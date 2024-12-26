from mmdet.models.backbones.dift_3.src.models.hyperfeature_3 import load_models_stride_hf_3
import torch
from torch import nn
import sys
import os
import torch
from torch import nn

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))


class HyperFeatureEncoder_3(nn.Module):
    def __init__(self, mode="float", dift_config=None):
        super().__init__()
        self.mode = mode
        self.diffusion_extractor, self.aggregation_network = load_models_stride_hf_3(
            dift_config)

    def forward(self, img_tensor):
        with torch.no_grad():
            _, timesteps_features = self.diffusion_extractor.forward(
                img_tensor)

        timesteps_features = self.aggregation_network.forward(
            timesteps_features)
        return timesteps_features
