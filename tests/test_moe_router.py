import pytest
import torch
from alps.core.moe_router import SparseMoERouter

def test_moe_router():
    """Validates Mixture of Experts routing and load balancing calculations."""
    d_model = 64
    router = SparseMoERouter(d_model=d_model, num_experts=4, active_experts=2, d_ff=128)
    
    inputs = torch.randn(2, 10, d_model)
    outputs, balance_loss = router(inputs)
    
    assert outputs.shape == inputs.shape
    assert balance_loss.item() > 0
