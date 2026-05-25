import torch
from alps.core.encoders import VisionEncoder

class VisionPrimaryEncoder(VisionEncoder):
    """
    Vision-Primary Encoder Module.
    Wraps the core 3D VisionTransformer to serve as our always-on primary modality.
    """
    def __init__(self, in_channels: int = 3, d_model: int = 384, depth: int = 8, 
                 num_heads: int = 6, mlp_ratio: float = 4.0, dropout: float = 0.1,
                 patch_size: tuple = (2, 16, 16), max_patches: int = 2048):
        super().__init__(
            in_channels=in_channels, d_model=d_model, depth=depth, 
            num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout,
            patch_size=patch_size, max_patches=max_patches
        )
