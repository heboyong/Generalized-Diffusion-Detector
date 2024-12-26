from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
import os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from archs.detectron2.resnet import BottleneckBlock

class DiftFinecoder_0(nn.Module):
    def __init__(self, in_channel=384, out_channels=[384, 384, 384, 384]):
        super().__init__()
        self.in_channel = in_channel
        self.out_channels = out_channels
        self.bn = nn.BatchNorm2d(in_channel)
        self.down = nn.MaxPool2d(kernel_size=2, stride=(2, 2))

        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)
        
    def forward(self, dift_values):
        dift_values = self.bn(dift_values)
        # Resolution High to Low
        highest_dift = F.interpolate(dift_values, scale_factor=2, mode='bilinear', align_corners=True)
        mid_dift = self.down(dift_values)
        low_dift = self.down(mid_dift)
        return highest_dift, dift_values, mid_dift, low_dift


class DiftFinecoder(nn.Module):
    def __init__(self, in_channel=384, out_channels=[96, 384, 384, 768]):
        super().__init__()
        self.in_channel = in_channel
        self.out_channels = out_channels
        self.bn = nn.BatchNorm2d(in_channel)
        self.down_0 = nn.Conv2d(in_channels=in_channel, out_channels=out_channels[2], kernel_size=2, stride=(2, 2))
        self.down_1 = nn.Conv2d(in_channels=out_channels[2], out_channels=out_channels[3], kernel_size=2, stride=(2, 2))
        self.up = nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channels[0], kernel_size=2, stride=(2, 2))

        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)
        nn.init.kaiming_normal_(self.down_0.weight)
        nn.init.zeros_(self.down_0.bias)
        nn.init.kaiming_normal_(self.down_1.weight)
        nn.init.zeros_(self.down_1.bias)
        nn.init.kaiming_normal_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        
    
    def forward(self, dift_values):
        dift_values = self.bn(dift_values)
        # Resolution High to Low
        highest_dift = self.up(dift_values)
        mid_dift = self.down_0(dift_values)
        low_dift = self.down_1(mid_dift)
        return highest_dift, dift_values, mid_dift, low_dift


class DiftUNetFinecoder(nn.Module):
    def __init__(self, in_channels=384, hidden_dim=[92, 384, 384, 768]):
        super(DiftUNetFinecoder, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim[2], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(hidden_dim[2], hidden_dim[3], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Multihead Attention
        # self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim[3], hidden_dim[2], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim[2], hidden_dim[1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim[1], hidden_dim[0], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # Encoding
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)

        # Decoding
        dec1_out = F.upsample_bilinear(self.dec1(enc2_out), size=(enc1_out.shape[2], enc1_out.shape[3])) + enc1_out
        dec2_out = self.dec2(dec1_out) + x

        # Upsample to match original resolution
        dec3_out = self.dec3(dec2_out)

        return dec3_out, dec2_out, dec1_out, enc2_out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=24):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.gn2 = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out += residual
        out = self.relu(out)
        return out

class EnhancedFeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels, num_group=24):
        super(EnhancedFeatureFusion, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.resblock = ResBlock(out_channels, out_channels)

    def forward(self, x1, x2=None):
        x1 = self.deconv(x1)
        if x2 is not None:
            x1 = x1 + x2
        x1 = self.resblock(x1)
        return x1

class EnhancedFusionNetwork(nn.Module):
    def __init__(self, in_channels=[768, 384, 192], out_channel=96):
        super(EnhancedFusionNetwork, self).__init__()
        self.fusion1 = EnhancedFeatureFusion(in_channels[0], in_channels[1])
        self.fusion2 = EnhancedFeatureFusion(in_channels[1], in_channels[2])
        self.fusion3 = EnhancedFeatureFusion(in_channels[2], out_channel)
        self.down1 = BottleneckBlock(in_channels=3, out_channels=48, bottleneck_channels=out_channel//4, stride=2, num_norm_groups=3)
        self.down2 = BottleneckBlock(in_channels=48, out_channels=96, bottleneck_channels=out_channel//4, stride=2, num_norm_groups=12)
        self.fusion_out = BottleneckBlock(in_channels=out_channel, out_channels=out_channel, bottleneck_channels=out_channel//4, num_norm_groups=24)

    def forward(self, x, x8, x16, x32):
        fusion_16 = self.fusion1(x32, x16)
        fusion_8 = self.fusion2(fusion_16, x8)
        fusion_4 = self.fusion3(fusion_8)
        
        down_2 = self.down1(x)
        down_4 = self.down2(down_2)
        
        out = down_4 + fusion_4
        out = self.fusion_out(out)
        return out

# class EnhancedFusionNetwork(nn.Module):
#     def __init__(self, in_channels=[768, 384, 192], out_channel=96):
#         super(EnhancedFusionNetwork, self).__init__()
#         self.fusion1 = EnhancedFeatureFusion(in_channels[0], in_channels[1])
#         self.fusion2 = EnhancedFeatureFusion(in_channels[1], in_channels[2])
#         self.fusion3 = EnhancedFeatureFusion(in_channels[2], out_channel)

#     def forward(self, x8, x16, x32):
#         x = self.fusion1(x32, x16)
#         x = self.fusion2(x, x8)
#         x = self.fusion3(x)
#         return x


# class DiftStrideFinecoder(nn.Module):
#     def __init__(self, in_channels=[768, 384, 192], hidden_dim_x4=96, num_group=24):
#         super().__init__()
#         self.fusion_net = EnhancedFusionNetwork(in_channels, hidden_dim_x4)
#         self.gn_x8 = nn.GroupNorm(num_group, in_channels[2])
#         self.gn_x16 = nn.GroupNorm(num_group, in_channels[1])
#         self.gn_x32 = nn.GroupNorm(num_group, in_channels[0])
#         # self.gn_x8 = nn.BatchNorm2d(in_channels[2])
#         # self.gn_x16 = nn.BatchNorm2d(in_channels[1])
#         # self.gn_x32 = nn.BatchNorm2d(in_channels[0])
#         self.apply(self.weights_init)

#     def forward(self, x, x8, x16, x32):
#         x8 = self.gn_x8(x8)
#         x16 = self.gn_x16(x16)
#         x32 = self.gn_x32(x32)
#         x4 = self.fusion_net(x, x8, x16, x32)
#         # x4 = self.fusion_net(x8, x16, x32)
#         return x4, x8, x16, x32
    
#     def weights_init(self, m):
#         """
#         初始化网络权重。
#         对于卷积层使用kaiming初始化，对于偏置使用0初始化。
#         """
#         if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#             init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             if m.bias is not None:
#                 init.constant_(m.bias, 0)
#         elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
#             init.constant_(m.weight, 1)
#             init.constant_(m.bias, 0)


class DiftStrideFinecoder(nn.Module):
    def __init__(self, in_channels=[768, 384, 192, 96], num_group=24):
        super().__init__()
        # self.gn_x8 = nn.GroupNorm(num_group, in_channels[3])
        # self.gn_x16 = nn.GroupNorm(num_group, in_channels[2])
        # self.gn_x32 = nn.GroupNorm(num_group, in_channels[1])
        # self.gn_x64 = nn.GroupNorm(num_group, in_channels[0])
        # self.apply(self.weights_init)

    def forward(self, x8, x16, x32, x64):
        # x8 = self.gn_x8(x8)
        # x16 = self.gn_x16(x16)
        # x32 = self.gn_x32(x32)
        # x64 = self.gn_x64(x64)

        x4_up = F.interpolate(x8, scale_factor=2, mode='bilinear', align_corners=True)
        x8_up = F.interpolate(x16, scale_factor=2, mode='bilinear', align_corners=True)
        x16_up = F.interpolate(x32, scale_factor=2, mode='bilinear', align_corners=True)
        x32_up = F.interpolate(x64, scale_factor=2, mode='bilinear', align_corners=True)

        return x4_up, x8_up, x16_up, x32_up

    def weights_init(self, m):
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)