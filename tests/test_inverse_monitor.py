import pytest
import torch
from alps.core.inverse_monitor import InverseMonitor

def test_inverse_monitor_interrupts():
    """Validates efference copy divergence calculations and interrupt triggers."""
    # Initialize monitor with threshold
    threshold = 0.5
    monitor = InverseMonitor(threshold=threshold)
    
    # 1. Test highly similar predictions (no interrupt expected)
    z_pred = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    z_target = torch.tensor([[[1.05, 0.0], [0.0, 0.95]]])
    
    div, interrupt = monitor(z_pred, z_target)
    assert div.item() < threshold
    assert interrupt is False
    
    # 2. Test divergent predictions (interrupt expected!)
    z_divergent = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    
    div_div, interrupt_div = monitor(z_pred, z_divergent)
    assert div_div.item() > threshold
    assert interrupt_div is True
