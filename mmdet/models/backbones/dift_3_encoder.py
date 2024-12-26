# Copyright (c) OpenMMLab. All rights reserved.

import torch
from mmengine.model import BaseModule

from mmdet.registry import MODELS
from .dift_3.src.models.encoder_hyperfeature_3 import HyperFeatureEncoder_3


@MODELS.register_module()
class DIFT_3(BaseModule):
    def __init__(self,
                 init_cfg=None,
                 dift_config=dict(
                     model_id="../stable-diffusion-3-medium-diffusers",
                     prompt="",
                     negative_prompt="",
                     save_timestep=5,
                     num_timesteps=50),
                 dift_type='HyperFeature'):
        super().__init__(init_cfg)

        self.dift_model = None
        assert dift_config is not None
        self.dift_config = dift_config
        if dift_type == 'HyperFeature':
            self.dift_model = HyperFeatureEncoder_3(dift_config=self.dift_config)

    def forward(self, x):
        x = self.imagenet_to_stable_diffusion(x)
        x = self.dift_model(x.to(dtype=torch.float16))
        return x

    def imagenet_to_stable_diffusion(self, tensor):
        mean = torch.tensor([123.675, 116.28, 103.53]).view(
            1, 3, 1, 1).to(tensor.device)
        std = torch.tensor([58.395, 57.12, 57.375]).view(
            1, 3, 1, 1).to(tensor.device)
        tensor = tensor * std + mean
        tensor = tensor / 255.0
        tensor = tensor * 2.0 - 1.0
        return tensor
