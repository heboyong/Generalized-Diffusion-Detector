import torch
from torch import nn
from .stable_diffusion.sd3_custom import CustomStableDiffusion3Img2ImgPipeline
from .stable_diffusion.sd3_transformer_custom import CustomSD3Transformer2DModel
import os


def freeze_weights(weights):
    for param in weights.parameters():
        param.requires_grad = False


class DiffusionExtractor(nn.Module):
    """
    Module for running either the generation or inversion process 
    and extracting intermediate feature maps.
    """

    def __init__(self, config, device):
        super().__init__()
        self.device = device

        transformer_path = os.path.join(config["model_id"], 'transformer')
        transformer = CustomSD3Transformer2DModel.from_pretrained(
            transformer_path, torch_dtype=torch.float16)

        self.pipe = CustomStableDiffusion3Img2ImgPipeline.from_pretrained(config["model_id"],
                                                                          transformer=transformer,
                                                                          text_encoder_3=None,
                                                                          tokenizer_3=None,
                                                                          torch_dtype=torch.float16)
        self.pipe = self.pipe.to(device)

        # Get settings
        self.num_timesteps = config["num_timesteps"]
        self.generator = torch.Generator(
            self.device).manual_seed(config.get("seed", 0))

        self.prompt = config.get("prompt", "")
        self.negative_prompt = config.get("negative_prompt", "")
        self.save_timestep = config.get("save_timestep", [])

        # freeze
        freeze_weights(self.pipe.text_encoder)
        freeze_weights(self.pipe.text_encoder_2)
        freeze_weights(self.pipe.vae)
        freeze_weights(self.pipe.transformer)

    def change_batchsize(self, batch_size):
        self.batch_size = batch_size

    def forward(self, images=None):
        if images is not None:
            output, timesteps_features = self.pipe(
                image=images,
                prompt=self.prompt,
                feature_timesteps=self.save_timestep,
                num_inference_steps=self.num_timesteps,
                negative_prompt=self.negative_prompt,
                generator=self.generator
            )
            h = images.shape[2]
            w = images.shape[3]

            timesteps_features = torch.stack(timesteps_features)
            timesteps_features = timesteps_features.permute(2, 0, 1, 4, 3)
            (N, timesteps, blocks, C), H, W = timesteps_features.shape[0:4], int(
                h//16), int(w//16)
            timesteps_features = timesteps_features.view(
                N, timesteps, blocks, C, H, W).contiguous()

        return output, timesteps_features
