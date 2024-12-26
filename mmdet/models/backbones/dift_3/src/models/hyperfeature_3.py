import os
import sys

# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ..archs.diffusion_extractor import DiffusionExtractor
from ..archs.aggregation_network import AggregationNetwork

def load_models_stride_hf_3(dift_config, device='cuda'):
    config = dift_config
    diffusion_extractor = DiffusionExtractor(config, device)
    aggregation_network = AggregationNetwork(
        feature_dim=1536, timesteps=config["save_timestep"])
    return diffusion_extractor, aggregation_network
