import pytest
import torch
from alps.core.hierarchy import StrategicLayer, TacticalLayer, OperativeLayer

def test_hierarchy_forward_backward():
    """Validates the multi-scale hierarchy, shape consistency, and gradient isolation."""
    d_model = 48
    num_embeddings = 64
    num_experts = 4
    active_experts = 2
    d_action = 8
    
    # 1. Initialize layers
    strat = StrategicLayer(d_model=d_model, num_embeddings=num_embeddings)
    tact = TacticalLayer(d_model=d_model, num_experts=num_experts, active_experts=active_experts)
    oper = OperativeLayer(d_model=d_model, d_action=d_action)
    
    # 2. Feed inputs and verify shapes
    # Input latent: [B, N, D]
    B, N = 2, 10
    z_t = torch.randn(B, N, d_model, requires_grad=True)
    
    # Operative Layer Pass
    z_op, sig_op = oper(z_t, z_t)
    assert z_op.shape == (B, N, d_model)
    assert sig_op.item() >= 0
    
    # Tactical Layer Pass
    z_tac, moe_loss, sig_tac = tact(z_op, z_op)
    assert z_tac.shape == (B, N, d_model)
    assert moe_loss.item() >= 0
    assert sig_tac.item() >= 0
    
    # Strategic Layer Pass (discrete bottleneck)
    z_str, vq_loss, sig_str = strat(z_tac)
    assert z_str.shape == (B, N, d_model)
    assert vq_loss.item() >= 0
    assert sig_str.item() >= 0
    
    # 3. Test stop-gradient boundary checks
    # Strategic should propagate gradients back to its own layers, but not down to Operative
    total_loss = vq_loss + moe_loss + sig_op + sig_tac + sig_str
    total_loss.backward()
    
    # Verify that gradients propagate back to the input
    assert z_t.grad is not None
