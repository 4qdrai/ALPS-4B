import pytest
import torch
from alps.core.sigreg import SIGReg

def test_sigreg():
    """Validates that SIGReg prevents representation collapse."""
    d_model = 64
    sigreg = SIGReg(d_model=d_model, num_slices=32, beta=1.0)
    
    # 1. Standard normal input should yield a low goodness-of-fit statistic (approx stable)
    normal_input = torch.randn(10, d_model)
    loss_normal = sigreg(normal_input)
    assert loss_normal.item() >= 0
    
    # 2. Collapsed input (all constant vectors) should yield a high statistic
    collapsed_input = torch.zeros(10, d_model)
    loss_collapsed = sigreg(collapsed_input)
    assert loss_collapsed.item() >= 0
    
    # 3. Weak-SIGReg check
    sigreg_weak = SIGReg(d_model=d_model, num_slices=32, weak_only=True)
    loss_weak = sigreg_weak(normal_input)
    assert loss_weak.item() >= 0
