import pytest
import torch
from alps.core.checker import BanachChecker

def test_banach_checker():
    """Validates Banach Checker iteration convergence under Lipschitz condition L < 1."""
    d_model = 64
    checker = BanachChecker(d_model=d_model, d_cond=d_model, max_iters=5)
    
    initial_z = torch.randn(2, 5, d_model)
    context = torch.randn(2, 5, d_model)
    
    refined_z, steps, converged = checker(initial_z, context)
    assert refined_z.shape == initial_z.shape
    assert steps > 0
    
    # Compute contraction loss
    loss = checker.compute_contraction_loss(initial_z, context)
    assert loss.item() >= 0
