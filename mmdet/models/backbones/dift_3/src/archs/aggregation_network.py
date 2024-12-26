import torch
from torch import nn
import torch.nn.init as init
from archs.detectron2.resnet import ResNet, BottleneckBlock


class AggregationNetwork(nn.Module):
    def __init__(
        self,
        # Input feature channel (C), assuming all feature dimensions are the same
        feature_dim: int = 1536,
        timesteps: int = 5,             # Number of timesteps
        blocks: int = 24,               # Number of blocks per timestep
        device: torch.device = 'cuda',
        projection_dim: int = 1536,
        num_norm_groups: int = 32,
        num_res_blocks: int = 1,
    ):
        super(AggregationNetwork, self).__init__()

        self.feature_dim = feature_dim
        self.timesteps = timesteps
        self.blocks = blocks
        self.device = device
        self.projection_dim = projection_dim

        # Initialize independent bottleneck layers for each timestep and block
        self.bottleneck_layers = nn.ModuleList()
        for t in range(timesteps):
            for b in range(blocks):
                bottleneck = nn.Sequential(
                    *ResNet.make_stage(
                        BottleneckBlock,
                        num_blocks=num_res_blocks,
                        in_channels=feature_dim,
                        bottleneck_channels=projection_dim // 8,
                        out_channels=projection_dim,
                        norm="GN",
                        num_norm_groups=num_norm_groups
                    )
                )
                self.bottleneck_layers.append(bottleneck)

        # Initialize mixing weights for timesteps and blocks
        self.timesteps_weights = nn.Parameter(
            torch.ones(timesteps))  # Shape: [timesteps]
        self.blocks_weights = nn.Parameter(
            torch.ones(blocks))        # Shape: [blocks]

        self.half()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Assume the input `batch` has the shape [N, timesteps, blocks, C, H, W].

        Returns aggregated features with shape [N, projection_dim, H, W].
        """
        N, T, B, C, H, W = batch.shape

        # Ensure the number of timesteps and blocks matches the initialization
        assert T == self.timesteps, f"Expected timesteps {self.timesteps}, but got {T}"
        assert B == self.blocks, f"Expected blocks {self.blocks}, but got {B}"

        # Initialize the output feature tensor to zeros
        output_feature = torch.zeros(
            N, self.projection_dim, H, W, device=self.device, dtype=self.timesteps_weights.dtype
        )

        # Apply softmax to timesteps and blocks weights
        timesteps_softmax = torch.softmax(
            self.timesteps_weights, dim=0)  # Shape: [timesteps]
        blocks_softmax = torch.softmax(
            self.blocks_weights, dim=0)        # Shape: [blocks]

        # print(timesteps_softmax.data, blocks_softmax.data)

        # Calculate combined weights: outer product to get [timesteps, blocks]
        combined_weights = timesteps_softmax.unsqueeze(
            1) * blocks_softmax.unsqueeze(0)  # Shape: [timesteps, blocks]

        # Iterate over each timestep
        for t in range(T):
            # Iterate over each block
            for b in range(B):
                # Get the current block's features, shape [N, C, H, W]
                current_block = batch[:, t, b, :, :, :]
                # Get the corresponding bottleneck layer
                bottleneck_layer = self.bottleneck_layers[t * B + b]
                # Pass through the independent bottleneck layer, shape becomes [N, projection_dim, H, W]
                bottlenecked_feature = bottleneck_layer(current_block)
                # Weight and accumulate into the output feature
                weight = combined_weights[t, b]
                output_feature += weight * bottlenecked_feature

        return output_feature
