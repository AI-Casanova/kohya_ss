import argparse
import torch
import os
try:
    import intel_extension_for_pytorch as ipex
    if torch.xpu.is_available():
        from library.ipex import ipex_init
        ipex_init()
except Exception:
    pass
from library import sdxl_model_util, sdxl_train_util, train_util
import train_network
import library.model_util as model_util


class SdxlNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.vae_scale_factor = sdxl_model_util.VAE_SCALE_FACTOR
        self.is_sdxl = True

    def assert_extra_args(self, args, train_dataset_group):
        super().assert_extra_args(args, train_dataset_group)
        sdxl_train_util.verify_sdxl_training_args(args)

        if args.cache_text_encoder_outputs:
            assert (
                train_dataset_group.is_text_encoder_output_cacheable()
            ), "when caching Text Encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / Text Encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

        assert (
            args.network_train_unet_only or not args.cache_text_encoder_outputs
        ), "network for Text Encoder cannot be trained with caching Text Encoder outputs / Text Encoderの出力をキャッシュしながらText Encoderのネットワークを学習することはできません"

        train_dataset_group.verify_bucket_reso_steps(32)

    def load_target_model(self, args, weight_dtype, accelerator):
        (
            load_stable_diffusion_format,
            text_encoder1,
            text_encoder2,
            vae,
            unet,
            logit_scale,
            ckpt_info,
        ) = sdxl_train_util.load_target_model(args, accelerator, sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, weight_dtype)

        self.load_stable_diffusion_format = load_stable_diffusion_format
        self.logit_scale = logit_scale
        self.ckpt_info = ckpt_info

        return sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, [text_encoder1, text_encoder2], vae, unet

    def load_tokenizer(self, args):
        tokenizer = sdxl_train_util.load_tokenizers(args)
        return tokenizer

    def load_textual_inversion(self, args, tokenizers, text_encoders):
        if args.textual_inversion_embeddings:
            token_ids_embeds1 = []
            token_ids_embeds2 = []
            for embeds_file in args.textual_inversion_embeddings:
                if model_util.is_safetensors(embeds_file):
                    from safetensors.torch import load_file
                    data = load_file(embeds_file)
                else:
                    data = torch.load(embeds_file, map_location="cpu")
                if "string_to_param" in data:
                    data = data["string_to_param"]
                embeds1 = data["clip_l"]  # text encoder 1
                embeds2 = data["clip_g"]  # text encoder 2
                num_vectors_per_token = embeds1.size()[0]
                token_string = args.textual_inversion_name or os.path.splitext(os.path.basename(embeds_file))[0]
                token_strings = [token_string] + [f"{token_string}{i + 1}" for i in range(num_vectors_per_token - 1)]
                # add new word to tokenizer, count is num_vectors_per_token
                num_added_tokens1 = tokenizers[0].add_tokens(token_strings)
                num_added_tokens2 = tokenizers[1].add_tokens(token_strings)
                assert num_added_tokens1 == num_vectors_per_token and num_added_tokens2 == num_vectors_per_token, (
                        f"tokenizer has same word to token string (filename): {embeds_file}"
                        + f" / 指定した名前（ファイル名）のトークンが既に存在します: {embeds_file}"
                )
                token_ids1 = tokenizers[0].convert_tokens_to_ids(token_strings)
                token_ids2 = tokenizers[1].convert_tokens_to_ids(token_strings)
                print(
                    f"Textual Inversion embeddings `{token_string}` loaded. Tokens are added: {token_ids1} and {token_ids2}")
                assert (
                        min(token_ids1) == token_ids1[0] and token_ids1[-1] == token_ids1[0] + len(token_ids1) - 1
                ), f"token ids1 is not ordered"
                assert (
                        min(token_ids2) == token_ids2[0] and token_ids2[-1] == token_ids2[0] + len(token_ids2) - 1
                ), f"token ids2 is not ordered"
                assert len(tokenizers[0]) - 1 == token_ids1[-1], f"token ids 1 is not end of tokenize: {len(tokenizers[0])}"
                assert len(tokenizers[1]) - 1 == token_ids2[-1], f"token ids 2 is not end of tokenize: {len(tokenizers[1])}"
                # replacing with tokenid expansion
                # if num_vectors_per_token > 1:
                #     pipe.add_token_replacement(0, token_ids1[0], token_ids1)  # hoge -> hoge, hogea, hogeb, ...
                #     pipe.add_token_replacement(1, token_ids2[0], token_ids2)
                token_ids_embeds1.append((token_ids1, embeds1))
                token_ids_embeds2.append((token_ids2, embeds2))
            text_encoders[0].resize_token_embeddings(len(tokenizers[0]))
            text_encoders[1].resize_token_embeddings(len(tokenizers[1]))
            token_embeds1 = text_encoders[0].get_input_embeddings().weight.data
            token_embeds2 = text_encoders[1].get_input_embeddings().weight.data
            for token_ids, embeds in token_ids_embeds1:
                for token_id, embed in zip(token_ids, embeds):
                    token_embeds1[token_id] = embed
            for token_ids, embeds in token_ids_embeds2:
                for token_id, embed in zip(token_ids, embeds):
                    token_embeds2[token_id] = embed

    def is_text_encoder_outputs_cached(self, args):
        return args.cache_text_encoder_outputs

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator, unet, vae, tokenizers, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                # メモリ消費を減らす
                print("move vae and unet to cpu to save memory")
                org_vae_device = vae.device
                org_unet_device = unet.device
                vae.to("cpu")
                unet.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # When TE is not be trained, it will not be prepared so we need to use explicit autocast
            with accelerator.autocast():
                dataset.cache_text_encoder_outputs(
                    tokenizers,
                    text_encoders,
                    accelerator.device,
                    weight_dtype,
                    args.cache_text_encoder_outputs_to_disk,
                    accelerator.is_main_process,
                )

            text_encoders[0].to("cpu", dtype=torch.float32)  # Text Encoder doesn't work with fp16 on CPU
            text_encoders[1].to("cpu", dtype=torch.float32)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not args.lowram:
                print("move vae and unet back to original device")
                vae.to(org_vae_device)
                unet.to(org_unet_device)
        else:
            # Text Encoderから毎回出力を取得するので、GPUに乗せておく
            text_encoders[0].to(accelerator.device)
            text_encoders[1].to(accelerator.device)

    def get_text_cond(self, args, accelerator, batch, tokenizers, text_encoders, weight_dtype):
        if "text_encoder_outputs1_list" not in batch or batch["text_encoder_outputs1_list"] is None:
            input_ids1 = batch["input_ids"]
            input_ids2 = batch["input_ids2"]
            with torch.enable_grad():
                # Get the text embedding for conditioning
                # TODO support weighted captions
                # if args.weighted_captions:
                #     encoder_hidden_states = get_weighted_text_embeddings(
                #         tokenizer,
                #         text_encoder,
                #         batch["captions"],
                #         accelerator.device,
                #         args.max_token_length // 75 if args.max_token_length else 1,
                #         clip_skip=args.clip_skip,
                #     )
                # else:
                input_ids1 = input_ids1.to(accelerator.device)
                input_ids2 = input_ids2.to(accelerator.device)
                encoder_hidden_states1, encoder_hidden_states2, pool2 = train_util.get_hidden_states_sdxl(
                    args.max_token_length,
                    input_ids1,
                    input_ids2,
                    tokenizers[0],
                    tokenizers[1],
                    text_encoders[0],
                    text_encoders[1],
                    None if not args.full_fp16 else weight_dtype,
                )
        else:
            encoder_hidden_states1 = batch["text_encoder_outputs1_list"].to(accelerator.device).to(weight_dtype)
            encoder_hidden_states2 = batch["text_encoder_outputs2_list"].to(accelerator.device).to(weight_dtype)
            pool2 = batch["text_encoder_pool2_list"].to(accelerator.device).to(weight_dtype)

            # # verify that the text encoder outputs are correct
            # ehs1, ehs2, p2 = train_util.get_hidden_states_sdxl(
            #     args.max_token_length,
            #     batch["input_ids"].to(text_encoders[0].device),
            #     batch["input_ids2"].to(text_encoders[0].device),
            #     tokenizers[0],
            #     tokenizers[1],
            #     text_encoders[0],
            #     text_encoders[1],
            #     None if not args.full_fp16 else weight_dtype,
            # )
            # b_size = encoder_hidden_states1.shape[0]
            # assert ((encoder_hidden_states1.to("cpu") - ehs1.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
            # assert ((encoder_hidden_states2.to("cpu") - ehs2.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
            # assert ((pool2.to("cpu") - p2.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
            # print("text encoder outputs verified")

        return encoder_hidden_states1, encoder_hidden_states2, pool2

    def call_unet(self, args, accelerator, unet, noisy_latents, timesteps, text_conds, batch, weight_dtype):
        noisy_latents = noisy_latents.to(weight_dtype)  # TODO check why noisy_latents is not weight_dtype

        # get size embeddings
        orig_size = batch["original_sizes_hw"]
        crop_size = batch["crop_top_lefts"]
        target_size = batch["target_sizes_hw"]
        embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)

        # concat embeddings
        encoder_hidden_states1, encoder_hidden_states2, pool2 = text_conds
        vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
        text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

        noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)
        return noise_pred

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        sdxl_train_util.sample_images(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet)


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    sdxl_train_util.add_sdxl_training_arguments(parser)
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)

    trainer = SdxlNetworkTrainer()
    trainer.train(args)
