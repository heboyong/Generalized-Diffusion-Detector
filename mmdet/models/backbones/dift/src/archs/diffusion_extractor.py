from PIL import Image
import torch
from torch import nn

from diffusers import DDIMScheduler
from archs.stable_diffusion.diffusion import (
    init_models,
    get_tokens_embedding,
    generalized_steps,
    collect_and_resize_feats,
    collect_stride_feats_with_timesteplist,
)
from archs.stable_diffusion.resnet import init_resnet_func

from transformers import BlipProcessor, BlipForConditionalGeneration


class DiffusionExtractor(nn.Module):
    """
    Module for running either the generation or inversion process 
    and extracting intermediate feature maps.
    """

    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )
        self.num_timesteps = config["num_timesteps"]
        self.scheduler.set_timesteps(self.num_timesteps)
        self.generator = torch.Generator(self.device).manual_seed(config.get("seed", 0))

        self.unet, self.vae, self.clip, self.clip_tokenizer = init_models(model_id=config["model_id"])
        self.prompt = config.get("prompt", "")
        self.negative_prompt = config.get("negative_prompt", "")

        self.batch_size = 8
        self.mode = 'train'
        self.batch_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.cond = {i: None for i in self.batch_list}
        self.uncond = {i: None for i in self.batch_list}
        self.set_cond(self.prompt, self.negative_prompt)
        del self.clip, self.clip_tokenizer

        self.diffusion_mode = config.get("diffusion_mode", "generation")
        if "idxs" in config and config["idxs"] is not None:
            self.idxs = config["idxs"]
        else:
            self.idxs = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]
        self.output_resolution = (config["input_resolution"][0] // 8, config["input_resolution"][1] // 8)

        # Note that save_timestep is in terms of number of generation steps
        # save_timestep = 0 is noise, save_timestep = T is a clean image
        # generation saves as [0...T], inversion saves as [T...0]
        self.save_timestep = config.get("save_timestep", [])

        if config.get("scheduler_timesteps", None):
            self.scheduler.timesteps = torch.Tensor(config["scheduler_timesteps"])

        print('timesteps', self.scheduler.timesteps)

        # self.blip_model = BlipForConditionalGeneration.from_pretrained(config.get("blip_model_id"), torch_dtype=torch.float16).to("cuda")

        # print(f"batch_size: {self.batch_size}")
        # print(f"diffusion_mode: {self.diffusion_mode}")
        # print(f"idxs: {self.idxs}")
        # print(f"output_resolution: {self.output_resolution}")
        # print(f"prompt: {self.prompt}")
        # print(f"negative_prompt: {self.snegative_prompt}")

    def set_cond(self, prompt, negative_prompt):
        with torch.no_grad():
            with torch.autocast("cuda"):
                _, cond_prompt = get_tokens_embedding(self.clip_tokenizer, self.clip, self.device, prompt)
                _, uncond_prompt = get_tokens_embedding(self.clip_tokenizer, self.clip, self.device, negative_prompt)
        for batch_size in self.batch_list:
            cond_tmp = cond_prompt.expand((batch_size, *cond_prompt.shape[1:]))
            cond_tmp = cond_tmp.to(self.device)

            uncond_tmp = uncond_prompt.expand((batch_size, *uncond_prompt.shape[1:]))
            uncond_tmp = uncond_tmp.to(self.device)

            self.cond[batch_size] = cond_tmp
            self.uncond[batch_size] = uncond_tmp

    def change_mode(self, mode='val'):
        self.mode = mode

    def change_batchsize(self, batch_size):
        self.batch_size = batch_size

    def change_cond(self, prompt, cond_type="cond", batch_size=2):
        with torch.no_grad():
            with torch.autocast("cuda"):
                _, new_cond = get_tokens_embedding(self.clip_tokenizer, self.clip, self.device, prompt)
                new_cond = new_cond.expand((batch_size, *new_cond.shape[1:]))
                new_cond = new_cond.to(self.device)
                if cond_type == "cond":
                    self.cond = new_cond
                    self.prompt = prompt
                elif cond_type == "uncond":
                    self.uncond = new_cond
                    self.negative_prompt = prompt
                else:
                    raise NotImplementedError

    def run_generation(self, latent, guidance_scale=-1, min_i=None, max_i=None):
        xs = generalized_steps(
            latent,
            self.unet,
            self.scheduler,
            run_inversion=False,
            guidance_scale=guidance_scale,
            conditional=self.cond,
            unconditional=self.uncond,
            min_i=min_i,
            max_i=max_i
        )
        return xs

    def run_inversion(self, latent, guidance_scale=-1, min_i=None, max_i=None):
        if self.mode == 'train':
            cond = self.cond[self.batch_size]
            uncond = self.uncond[self.batch_size]
        else:
            cond = self.cond[1]
            uncond = self.uncond[1]
        xs = generalized_steps(
            latent,
            self.unet,
            self.scheduler,
            run_inversion=True,
            guidance_scale=guidance_scale,
            conditional=cond,
            unconditional=uncond,
            min_i=min_i,
            max_i=max_i
        )
        return xs

    def get_feats(self, latents, extractor_fn, preview_mode=False):
        # returns feats of shape [batch_size, num_timesteps, channels, w, h]
        if not preview_mode:
            init_resnet_func(self.unet, save_hidden=True, reset=True, idxs=self.idxs, save_timestep=self.save_timestep)
        outputs = extractor_fn(latents)
        if not preview_mode:
            feats = []
            for timestep in self.save_timestep:
                timestep_feats = collect_and_resize_feats(self.unet, self.idxs, timestep, self.output_resolution)
                feats.append(timestep_feats)
            feats = torch.stack(feats, dim=1)
            init_resnet_func(self.unet, reset=True)
        else:
            feats = None
        return feats, outputs

    def get_feats_stride(self, latents, extractor_fn, preview_mode=False):
        # returns feats of shape [batch_size, num_timesteps, channels, w, h]
        if not preview_mode:
            init_resnet_func(self.unet, save_hidden=True, reset=True, idxs=self.idxs, save_timestep=self.save_timestep)
        outputs = extractor_fn(latents)
        if not preview_mode:
            feats = collect_stride_feats_with_timesteplist(self.unet, self.idxs, timestep_list=self.save_timestep)
            # feats = torch.stack(feats, dim=1)
            init_resnet_func(self.unet, reset=True)
        else:
            feats = None
        return feats, outputs

    def latents_to_images(self, latents):
        latents = latents.to(self.device)
        latents = latents / 0.18215
        images = self.vae.decode(latents.to(self.vae.dtype)).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")
        return [Image.fromarray(image) for image in images]

    def forward(self, images=None, latents=None, guidance_scale=-1, preview_mode=False, stride_mode=False):
        if images is None:
            if latents is None:
                latents = torch.randn((self.batch_size, self.unet.in_channels, 512 // 8, 512 // 8), device=self.device,
                                      generator=self.generator)
            if self.diffusion_mode == "generation":
                if preview_mode:
                    extractor_fn = lambda latents: self.run_generation(latents, guidance_scale, max_i=self.end_timestep)
                else:
                    extractor_fn = lambda latents: self.run_generation(latents, guidance_scale)
            elif self.diffusion_mode == "inversion":
                raise NotImplementedError
        else:
            # images = torch.nn.functional.interpolate(images, size=512, mode="bilinear")
            latents = self.vae.encode(images).latent_dist.sample(generator=None) * 0.18215

            # Run BLIP
            # caption_prompt = self.blip_model.generate({"image": images})
            # batch_size_tmp = images.shape[0]
            # self.change_cond(caption_prompt, cond_type="cond", batch_size=batch_size_tmp)

            # print('jyxjyxjyx vae latents', torch.isnan(latents).float().sum())
            if self.diffusion_mode == "inversion":
                extractor_fn = lambda latents: self.run_inversion(latents, guidance_scale)
            elif self.diffusion_mode == "generation":
                raise NotImplementedError

        with torch.no_grad():
            with torch.autocast("cuda"):
                if stride_mode:
                    return self.get_feats_stride(latents, extractor_fn)
                else:
                    return self.get_feats(latents, extractor_fn, preview_mode=preview_mode)
