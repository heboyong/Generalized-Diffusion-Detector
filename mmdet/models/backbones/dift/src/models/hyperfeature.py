import argparse
import glob
import json
import os
from omegaconf import OmegaConf
from PIL import Image
import random
import torch
from torch import nn
from tqdm import tqdm
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from archs.diffusion_extractor import DiffusionExtractor
from archs.aggregation_network import AggregationNetwork, StrideAggregationNetwork
from archs.stable_diffusion.resnet import collect_dims


def load_models(config_path, device="cuda"):
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    diffusion_extractor = DiffusionExtractor(config, device)
    dims = collect_dims(diffusion_extractor.unet, idxs=diffusion_extractor.idxs)

    aggregation_network = AggregationNetwork(
        projection_dim=config["projection_dim"],
        feature_dims=dims,
        device=device,
        save_timestep=config["save_timestep"],
        num_timesteps=config["num_timesteps"]
    )
    aggregation_network.load_state_dict(torch.load(config["weights_path"], map_location="cpu")["aggregation_network"])
    aggregation_network.half()
    return config, diffusion_extractor, aggregation_network


def load_models_stride_hf(dift_config, device='cuda'):
    config = dift_config
    # config = OmegaConf.create(dift_config)
    # config = OmegaConf.to_container(config, resolve=True)
    diffusion_extractor = DiffusionExtractor(config, device)
    dims = collect_dims(diffusion_extractor.unet, idxs=diffusion_extractor.idxs)
    aggregation_network = StrideAggregationNetwork(
        projection_dim=config["projection_dim"],
        feature_dims=dims,
        idxs=diffusion_extractor.idxs,
        device=device,
        save_timestep=config["save_timestep"],
        num_timesteps=config["num_timesteps"]
    )
    return config, diffusion_extractor, aggregation_network


class HyperFeaturizeBatch:
    def __init__(self, rank=0, config_path='/home/xmuairmud/jyx/DIFT-DA-Seg/configs/hyperfeature.yaml'):
        self.config, self.diffusion_extractor, self.aggregation_network = load_models(config_path, device=rank)
        self.rank = rank
        self.h, self.w = self.config['input_resolution'][0]//8, self.config['input_resolution'][1]//8

    @torch.no_grad()
    def forward(self, 
                img_tensor
                ):
        # img_tensor = img_tensor.to(self.rank, dtype=torch.float16)
        img_tensor = img_tensor
        b = img_tensor.shape[0]
        feats, _ = self.diffusion_extractor.forward(img_tensor)
        print(feats.shape)
        diffusion_hyperfeats = self.aggregation_network(feats.view((b, -1, self.h, self.w)))
        return diffusion_hyperfeats


class StrideHyperFeatureBatch(nn.Module):
    def __init__(self, rank=0, config_path='/home/xmuairmud/jyx/DIFT-DA-Seg/configs/stridehyperfeature.yaml'):
        super().__init__()
        self.config, self.diffusion_extractor, self.aggregation_network = load_models_stride_hf(config_path, device=rank)
        self.rank = rank

    def forward(self, 
                img_tensor
                ):
        # img_tensor = img_tensor.to(self.rank, dtype=torch.float16)
        img_tensor = img_tensor
        print(img_tensor.shape)
        b = img_tensor.shape[0]
        h = img_tensor.shape[2]
        w = img_tensor.shape[3]
        with torch.no_grad():
            feats, _ = self.diffusion_extractor.forward(img_tensor, stride_mode=True)
        print(feats[0].shape, feats[1].shape, feats[2].shape)
        diffusion_hyperfeats = self.aggregation_network([feats[0].view((b, -1, h//32, w//32)), feats[1].view((b, -1, h//16, w//16)), feats[2].view((b, -1, h//8, w//8))])
        return diffusion_hyperfeats 


if __name__ == '__main__':
    import cv2
    image = cv2.imread('/home/xmuairmud/jyx/data/GTA/images/24966.png')
    image = torch.tensor(image).unsqueeze(0).permute((0, 3, 1, 2)).to(dtype=torch.float)
    # print(image.shape)
    hf = HyperFeaturizeBatch(config_path='/home/xmuairmud/jyx/DIFT-DA-Seg/configs/stridehyperfeature.yaml')
    out = hf.forward(image)