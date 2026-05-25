import pytest
import torch
from alps.core.fallback import FallbackMonitor

def test_fallback_watchdog_and_lyapunov():
    """Validates the Fallback Monitor watchdogs and Lyapunov stable braking convergence."""
    d_model = 16
    monitor = FallbackMonitor(var_threshold=1e-3, pinning_threshold=0.99)
    
    # 1. Verify healthy system
    z_t = torch.randn(2, 5, d_model)
    z_prev = torch.randn(2, 5, d_model)
    healthy, msg = monitor.verify_system_health(z_t, z_prev)
    assert healthy is True
    assert msg == "SYSTEM_HEALTHY"
    
    # 2. Verify NaN trigger
    nan_z = z_t.clone()
    nan_z[0, 0, 0] = float("nan")
    healthy, msg = monitor.verify_system_health(nan_z, z_prev)
    assert healthy is False
    assert "NaN" in msg
    
    # 3. Verify Variance Collapse trigger
    collapsed_z = torch.zeros(2, 5, d_model)
    healthy, msg = monitor.verify_system_health(collapsed_z, z_prev)
    assert healthy is False
    assert "Variance" in msg
    
    # 4. Verify Pinning trigger
    pinned_z = z_t.clone()
    healthy, msg = monitor.verify_system_health(pinned_z, pinned_z)
    assert healthy is False
    assert "Pinning" in msg
    
    # 5. Verify Lyapunov stable braking step
    # Start at a high-velocity physical state
    x_state = torch.tensor([[10.0, 5.0]], dtype=torch.float32)
    
    # Perform 5 MRC braking steps and check Lyapunov energy convergence
    V_prev = None
    for step in range(5):
        next_state, V_val, V_dot = monitor.simulate_mrc_step(x_state, dt=0.05)
        
        # Verify Lyapunov candidate is strictly positive for non-zero states
        assert V_val.item() > 0.0
        
        # Verify Lyapunov derivative is strictly negative (energy decreases!)
        assert V_dot.item() < 0.0
        
        # Verify monotonic decrease: V_new < V_prev
        if V_prev is not None:
            assert V_val.item() < V_prev
            
        V_prev = V_val.item()
        x_state = next_state
        
    # State should have decreased towards origin [0, 0]
    assert torch.norm(x_state).item() < 12.0
