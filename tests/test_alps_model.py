import pytest
import torch
from alps.core.alps_model import ALPSModel

def test_alps_model_micro_pipeline():
    """
    Validates end-to-end forward/backward pipeline of ALPSModel.
    Uses a micro-configuration to ensure immediate execution on CPU/GPU.
    """
    d_model = 48
    d_action = 8
    
    # 1. Initialize ALPSModel with micro-scale parameters for fast testing
    model = ALPSModel(
        d_model=d_model,
        d_action=d_action,
        encoder_depth=1,
        encoder_num_heads=6,
        encoder_patch_size=(2, 8, 8),
        encoder_max_patches=64,
        use_langevin=True,
        langevin_steps=2
    )
    
    # 2. Input video tensor: [B, C, T, H, W] -> 4 frames at 16x16 resolution
    # Actions tensor: [B, D_action]
    B, C, T, H, W = 1, 3, 4, 16, 16
    video = torch.randn(B, C, T, H, W)
    actions = torch.randn(B, d_action)
    
    # 3. Forward pass
    outputs = model(video, actions)
    
    assert "z_t" in outputs
    assert "z_pred" in outputs
    assert "loss" in outputs
    assert "system_healthy" in outputs
    assert outputs["system_healthy"] is True
    assert outputs["fallback_triggered"] is False
    assert outputs["loss"].item() >= 0
    assert "refined_actions" in outputs
    assert outputs["refined_actions"].shape == (B, d_action)
    
    # 4. Backward pass
    outputs["loss"].backward()
    
    # Verify that gradients are computed successfully
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            # At least some parameters must have gradients
            break
    else:
        pytest.fail("No parameters computed gradients during backward pass.")
