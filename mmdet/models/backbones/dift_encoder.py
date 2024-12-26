# Copyright (c) OpenMMLab. All rights reserved.

import torch
from mmengine.model import BaseModule

from mmdet.registry import MODELS
from .dift.src.models.encoder_hyperfeature import HyperFeatureEncoder


@MODELS.register_module()
class DIFT(BaseModule):
    def __init__(self,
                 init_cfg=None,
                 dift_config=dict(projection_dim=[2048, 1024, 512, 256],
                                  projection_dim_x4=256,
                                  model_id="../stable-diffusion-v1-5",
                                  diffusion_mode="inversion",
                                  input_resolution=[512, 512],
                                  prompt="",
                                  negative_prompt="",
                                  guidance_scale=-1,
                                  scheduler_timesteps=[80, 60, 40, 20, 1],
                                  save_timestep=[4, 3, 2, 1, 0],
                                  num_timesteps=5,
                                  idxs=[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [3, 0],
                                        [3, 1],
                                        [3, 2]]),
                 dift_type='HyperFeature'):
        super().__init__(init_cfg)

        self.dift_model = None
        assert dift_config is not None
        self.dift_config = dift_config
        if dift_type == 'HyperFeature':
            self.dift_model = HyperFeatureEncoder(dift_config=self.dift_config)

    def forward(self, x):
        x = self.imagenet_to_stable_diffusion(x)
        x = self.dift_model(x.to(dtype=torch.float16))
        return x

    def init_weights(self):
        pass

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        pass

    def imagenet_to_stable_diffusion(self, tensor):
        # ImageNet 的均值和标准差
        mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(tensor.device)
        std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(tensor.device)

        # 逆标准化：将张量从 ImageNet 格式恢复到 [0, 255] 范围
        tensor = tensor * std + mean

        # 转换到 [0, 1] 范围
        tensor = tensor / 255.0

        # 转换到 [-1, 1] 范围
        tensor = tensor * 2.0 - 1.0

        return tensor
