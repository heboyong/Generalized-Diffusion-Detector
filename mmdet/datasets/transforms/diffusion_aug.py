import os

import numpy
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
import torch
import albumentations as A
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline


@TRANSFORMS.register_module()
class DiffusionAug(BaseTransform):

    def __init__(self,
                 prompt='random image style',
                 strength=0.5,
                 guidance_scale=5.0
                 ) -> None:
        self.prompt = prompt
        self.strength = strength
        self.guidance_scale = guidance_scale
        self.sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("/home/hby/project/stable-diffusion-v1-5",
                                                                      torch_dtype=torch.float16).to('cuda')

    def transform(self, results: dict) -> dict:
        image = self.sd_pipe(prompt=self.prompt, image=Image.fromarray(results['img']), strength=self.strength,
                             guidance_scale=self.guidance_scale).images[0]
        results['img'] = np.asarray(image)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'prompt={self.prompt}, '
        repr_str += f'strength={self.strength}, '
        repr_str += f'guidance_scale={self.guidance_scale})'
        return repr_str
