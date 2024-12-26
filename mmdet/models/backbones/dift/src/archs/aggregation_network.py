import numpy as np
import torch
from torch import nn
import torch.nn.init as init
from archs.detectron2.resnet import ResNet, BottleneckBlock


class StrideAggregationNetwork(nn.Module):
    """
    Module for aggreagating feature maps across time for diffrent strides (8, 16, 32).
    """
    def __init__(
            self, 
            feature_dims, 
            device,
            idxs,
            projection_dim=[768, 384, 192, 96],  ## Stride 64, 32, 16, 8
            num_norm_groups=32,
            num_res_blocks=1,
            save_timestep=[],
            num_timesteps=None,
            timestep_weight_sharing=False
        ):
        super().__init__()
        self.bottleneck_layers = nn.ModuleList()
        self.feature_dims = feature_dims
        # For CLIP symmetric cross entropy loss during training
        self.logit_scale = torch.ones([]) * np.log(1 / 0.07)
        self.device = device
        self.save_timestep = save_timestep

        # Count features for each stride, stride id [0, 1, 2]
        self.num_stride = len(projection_dim)
        self.feature_cnts = [0 for _ in range(self.num_stride)]
        self.feature_stride_idx = []
        self.feature_instride_num = []
        for i in range(len(idxs)):
            self.feature_stride_idx.append(idxs[i][0])
            self.feature_instride_num.append(self.feature_cnts[idxs[i][0]])
            self.feature_cnts[idxs[i][0]] += 1

        self.mixing_weights = []
        # self.mixing_weights_names = []
        for l, feature_dim in enumerate(self.feature_dims):
            stride_id = self.feature_stride_idx[l]

            bottleneck_layer = nn.Sequential(
                *ResNet.make_stage(
                    BottleneckBlock,
                    num_blocks=num_res_blocks,
                    in_channels=feature_dim,
                    bottleneck_channels=projection_dim[stride_id] // 4,
                    out_channels=projection_dim[stride_id],
                    norm="GN",
                    num_norm_groups=num_norm_groups
                )
            )
            self.bottleneck_layers.append(bottleneck_layer)
        
        self.bottleneck_layers = self.bottleneck_layers.to(device)
        mixing_weights = [torch.ones(self.feature_cnts[i] * len(save_timestep)) * 0.0001 for i in range(self.num_stride)]
        self.mixing_weights_stride = nn.ParameterList([nn.Parameter(mixing_weights[i].to(device)) for i in range(self.num_stride)])

        self.apply(self.weights_init)

    def weights_init(self, m):
        """
        初始化网络权重。
        对于卷积层使用kaiming初始化，对于偏置使用0初始化。
        """
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def forward(self, batch_list):
        """
        Assumes batch is a list of shape (B, C, H, W) where C is the concatentation of all timesteps features for each layer.

        Features are formed as (1,0)_timestep0,  (1,0)_timestep1, ..., (1,1)_timestep0, (1,1)_timestep1, ... 
        
        Return four features in stride 8, 16, 32, 64
        """

        output_features = [None for _ in range(self.num_stride)]
        mixing_weights_stride = [torch.nn.functional.softmax(self.mixing_weights_stride[i], dim=0) for i in range(self.num_stride)]
        # print(mixing_weights_stride[0], mixing_weights_stride[1], mixing_weights_stride[2])

        for idx_i in range(len(self.feature_stride_idx)):
            for timestep_i in range(len(self.save_timestep)):
                stride_id = self.feature_stride_idx[idx_i]
                instride_num = self.feature_instride_num[idx_i]

                # Share bottleneck layers across timesteps
                bottleneck_layer = self.bottleneck_layers[idx_i]
                # Chunk the batch according the layer
                # Account for looping if there are multiple timesteps
                num_channel = self.feature_dims[idx_i]
                start_channel = instride_num * len(self.save_timestep) * num_channel + timestep_i * num_channel
                feats = batch_list[stride_id][:, start_channel:start_channel+num_channel, :, :]
                # print(batch_list[stride_id].shape)
                # print(idx_i, f'timestep_i={timestep_i}, stride_id={stride_id}, instride_num={instride_num}, l={start_channel}, r={start_channel+num_channel}')
                # print(timestep_i * self.feature_cnts[stride_id] + instride_num)
                # Downsample the number of channels and weight the layer
                bottlenecked_feature = bottleneck_layer(feats)
                bottlenecked_feature = mixing_weights_stride[stride_id][timestep_i * self.feature_cnts[stride_id] + instride_num] * bottlenecked_feature
                if output_features[stride_id] is None:
                    output_features[stride_id] = bottlenecked_feature
                else:
                    output_features[stride_id] += bottlenecked_feature

        return output_features[3], output_features[2], output_features[1], output_features[0]


class AggregationNetwork(nn.Module):
    """
    Module for aggregating feature maps across time and space.
    Design inspired by the Feature Extractor from ODISE (Xu et. al., CVPR 2023).
    https://github.com/NVlabs/ODISE/blob/5836c0adfcd8d7fd1f8016ff5604d4a31dd3b145/odise/modeling/backbone/feature_extractor.py
    """
    def __init__(
            self, 
            feature_dims, 
            device, 
            projection_dim=384, 
            num_norm_groups=32,
            num_res_blocks=1, 
            save_timestep=[],
            num_timesteps=None,
            timestep_weight_sharing=False
        ):
        super().__init__()
        self.bottleneck_layers = nn.ModuleList()
        self.feature_dims = feature_dims    
        # For CLIP symmetric cross entropy loss during training
        self.logit_scale = torch.ones([]) * np.log(1 / 0.07)
        self.device = device
        self.save_timestep = save_timestep

        self.mixing_weights_names = []
        for l, feature_dim in enumerate(self.feature_dims):
            bottleneck_layer = nn.Sequential(
                *ResNet.make_stage(
                    BottleneckBlock,
                    num_blocks=num_res_blocks,
                    in_channels=feature_dim,
                    bottleneck_channels=projection_dim // 4,
                    out_channels=projection_dim,
                    norm="GN",
                    num_norm_groups=num_norm_groups
                )
            )
            self.bottleneck_layers.append(bottleneck_layer)
            for t in save_timestep:
                # 1-index the layer name following prior work
                self.mixing_weights_names.append(f"timestep-{save_timestep}_layer-{l+1}")
        
        self.bottleneck_layers = self.bottleneck_layers.to(device)
        mixing_weights = torch.ones(len(self.bottleneck_layers) * len(save_timestep))
        self.mixing_weights = nn.Parameter(mixing_weights.to(device))

    def forward(self, batch):
        """
        Assumes batch is shape (B, C, H, W) where C is the concatentation of all layer features.
        """
        output_feature = None
        start = 0
        mixing_weights = torch.nn.functional.softmax(self.mixing_weights)
        for i in range(len(mixing_weights)):
            # Share bottleneck layers across timesteps
            bottleneck_layer = self.bottleneck_layers[i % len(self.feature_dims)]
            # Chunk the batch according the layer
            # Account for looping if there are multiple timesteps
            end = start + self.feature_dims[i % len(self.feature_dims)]
            feats = batch[:, start:end, :, :]
            start = end
            # Downsample the number of channels and weight the layer
            bottlenecked_feature = bottleneck_layer(feats)
            bottlenecked_feature = mixing_weights[i] * bottlenecked_feature
            if output_feature is None:
                output_feature = bottlenecked_feature
            else:
                output_feature += bottlenecked_feature
        return output_feature