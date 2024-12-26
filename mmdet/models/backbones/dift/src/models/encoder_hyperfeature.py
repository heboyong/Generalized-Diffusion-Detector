import torch
from torch import nn
import sys
import os
import torch
from torch import nn

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))

from hyperfeature import load_models_stride_hf
from finecoder import DiftStrideFinecoder


class HyperFeatureEncoder(nn.Module):
    def __init__(self, batch_size=2, mode="float", dift_config=None):
        super().__init__()
        self.mode = mode

        # config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dift_config)

        self.config, self.diffusion_extractor, self.aggregation_network = load_models_stride_hf(dift_config)

        in_channels = self.config['projection_dim']
        hidden_dim_x4 = self.config['projection_dim_x4']

        self.finecoder = DiftStrideFinecoder(in_channels=in_channels)

        self.batch_size = batch_size

        self.change_batchsize(batch_size)

    def change_batchsize(self, batch_size):
        self.batch_size = batch_size
        self.diffusion_extractor.change_batchsize(batch_size)

    def change_mode(self, mode):
        self.diffusion_extractor.change_mode(mode)

    def change_precision(self, mode):
        self.mode = mode
        if mode == "float":
            self.aggregation_network.to(dtype=torch.float)
            self.finecoder.to(dtype=torch.float)
        elif mode == "half":
            self.aggregation_network.to(dtype=torch.float16)
            self.finecoder.to(dtype=torch.float16)

    def forward(self, img_tensor):
        b = img_tensor.shape[0]
        h = img_tensor.shape[2]
        w = img_tensor.shape[3]

        if b != self.batch_size:
            self.change_batchsize(b)

        with torch.no_grad():
            feats, _ = self.diffusion_extractor.forward(img_tensor, stride_mode=True)

        if self.mode == "float":
            stride_hf = self.aggregation_network([feats[0].view((b, -1, h // 64, w // 64)).to(dtype=torch.float),
                                                  feats[1].view((b, -1, h // 32, w // 32)).to(dtype=torch.float),
                                                  feats[2].view((b, -1, h // 16, w // 16)).to(dtype=torch.float),
                                                  feats[3].view((b, -1, h // 8, w // 8)).to(dtype=torch.float)])
        elif self.mode == "half":
            stride_hf = self.aggregation_network([feats[0].view((b, -1, h // 64, w // 64)),
                                                  feats[1].view((b, -1, h // 32, w // 32)),
                                                  feats[2].view((b, -1, h // 16, w // 16)),
                                                  feats[3].view((b, -1, h // 8, w // 8))])

        feature_fine = self.finecoder(stride_hf[0], stride_hf[1], stride_hf[2], stride_hf[3])

        return feature_fine

# if __name__ == '__main__':
#     from torchsummary import summary
#     encoder = HyperFeatureEncoder(mode='half')
#     encoder.to(0)
#     encoder.to(dtype=torch.float16)
#
#     img = cv2.imread('/home/xmuairmud/jyx/data/GTA/images/images/24966.png')
#
#     img = cv2.resize(img, dsize=(512, 512))
#     img_tensor = torch.Tensor(img)
#     img_tensor = img_tensor.permute(2, 0, 1)
#
#     img_tensor = img_tensor.to(device=0, dtype=torch.float16)
#     img_tensor = img_tensor[None, ...]
#     print(img_tensor.shape)
#
#     summary(encoder, input_size=(3, 512, 512))
#     # print(x[0].shape, x[1].shape, x[2].shape, x[3].shape)
