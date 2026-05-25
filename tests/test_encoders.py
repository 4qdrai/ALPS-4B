import pytest
import torch
from alps.core.encoders import VisionEncoder

def test_encoders():
    """Validates the spatiotemporal Vision Transformer Encoder and Patch Embedding."""
    B, C, T, H, W = 2, 3, 4, 32, 32
    d_model = 64
    
    encoder = VisionEncoder(
        in_channels=C, d_model=d_model, depth=2, num_heads=2, 
        patch_size=(2, 8, 8), max_patches=256
    )
    
    video = torch.randn(B, C, T, H, W)
    latents = encoder(video)
    
    # Expected output: [B, N, D]
    # T_out = 4 / 2 = 2
    # H_out = 32 / 8 = 4
    # W_out = 32 / 8 = 4
    # N = 2 * 4 * 4 = 32 patches
    assert latents.shape == (B, 32, d_model)
