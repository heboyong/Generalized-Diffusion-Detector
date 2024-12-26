from PIL import Image
import random
import torch
from torch import nn
from tqdm import tqdm
import sys
import os
import cv2
from transformers import Mask2FormerConfig, SegformerConfig
from torch.utils.data import Dataset, DataLoader
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))

from encoder_hyperfeature import HyperFeatureEncoder

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from decoder.mask2former.image_processing_dift_mask2former import DiftMask2FormerImageProcessor
from decoder.mask2former.modeling_dift_mask2former_0 import DiftMask2FormerForUniversalSegmentation_0
from decoder.segformer.modeling_dift_segformer import DiftSegformerForSemanticSegmentation
from utils.common import DataWithLabel


class HyperFeatureModel(nn.Module):
    def __init__(self, processor, batch_size=1, rank=0, config_path='/home/xmuairmud/jyx/DIFT-DA-Seg/configs/stridehyperfeature.yaml'):
        super().__init__()
        self.rank = rank
        self.encoder = HyperFeatureEncoder(rank=rank, config_path=config_path)
        # self.encoder.to(dtype=torch.float16)
        config = Mask2FormerConfig.from_pretrained('/home/xmuairmud/jyx/DIFT-DA-Seg/decoder/mask2former/diftmask2former-cityscapes-semantic')
        self.processor = processor
        self.decoder = DiftMask2FormerForUniversalSegmentation_0(config)
        self.change_batchsize(batch_size)
        self.encoder.to(rank)
        self.decoder.to(rank)

    def forward(self, x, labels=None):
        x = self.encoder(x)
        inputs = self.processor.preprocess(x, labels)
        x = self.decoder(**inputs)
        return x

    def change_batchsize(self, batch_size):
        self.encoder.diffusion_extractor.change_batchsize(batch_size)

    def change_precision(self, mode):
        self.encoder.change_precision(mode)
        if mode == "half":
            self.encoder.aggregation_network.to(dtype=torch.float16)
            self.encoder.finecoder.to(dtype=torch.float16)
            self.decoder.to(dtype=torch.float16)
        elif mode == "float":
            self.encoder.aggregation_network.to(dtype=torch.float)
            self.encoder.finecoder.to(dtype=torch.float)
            self.decoder.to(dtype=torch.float)


if __name__ == '__main__':
    config = Mask2FormerConfig.from_pretrained('../../../decoder/mask2former/diftmask2former-cityscapes-semantic')
    processor = DiftMask2FormerImageProcessor(ignore_index=255, device=0)
    model = HyperFeatureModel(processor=processor)

    image_dir = '/home/xmuairmud/jyx/data/GTA/images/images_train'
    label_dir = '/home/xmuairmud/jyx/data/GTA/labels/labels_train'

    dataset = DataWithLabel(image_dir, label_dir, output_size=(512, 512))
    loader = DataLoader(dataset, batch_size=1)

    img_tensor, label, _ = dataset[0]

    img_tensor = img_tensor[None, ...]
    label = np.array(label)
    label = label[None, ...]
    print(img_tensor.shape)
    print(label.shape)

    img_tensor = img_tensor.to(0)
    x = model(img_tensor, label)
    print(x)