import torch
import torch.nn as nn
import torch.nn.functional as F

class FallbackMonitor(nn.Module):
    """
    Fallback Watchdog (Brainstem/OOD Reflex Layer).
    
    Operates outside the gradient flow to monitor the mathematical health of the network.
    Since collapsed JEPA models output constants and yield zero prediction error (falsely
    reporting perfect performance), an independent watchdog is mandatory.
    
    Triggers:
    1. NaN / Infinity detection in latent variables.
    2. Variance Collapse: Var(z) < epsilon.
    3. Hypersphere Pinning: cos(z_t, z_{t-1}) >= 1 - delta (static freeze).
    
    On trigger, it immediately bypasses all neural inferences and executes
    the Minimal Risk Condition (MRC) to protect physical hardware.
    """
    def __init__(self, var_threshold: float = 1e-4, pinning_threshold: float = 0.999):
        super().__init__()
        self.var_threshold = var_threshold
        self.pinning_threshold = pinning_threshold
        
        # --- Control-Theoretic Lyapunov-Stable Braking Parameters ---
        # State vector: K=2 (position, velocity), Input: P=1 (braking effort)
        # Drift matrix A:
        self.register_buffer("A", torch.tensor([[0.0, 1.0], [0.0, -0.5]]))
        # Input coupling matrix B:
        self.register_buffer("B", torch.tensor([[0.0], [1.0]]))
        # Stabilizing controller feedback gain matrix K_gain:
        self.register_buffer("K_gain", torch.tensor([[1.0, 2.0]]))
        # Positive Definite Lyapunov matrix P satisfying Lyapunov Equation:
        self.register_buffer("P", torch.tensor([[2.25, 0.5], [0.5, 0.5]]))
        
    def check_nan_inf(self, z: torch.Tensor) -> bool:
        """Checks if there are any NaNs or Infinite values in the tensor."""
        if torch.isnan(z).any() or torch.isinf(z).any():
            return True
        return False
        
    def check_variance_collapse(self, z: torch.Tensor) -> bool:
        """
        Checks if the variance across the batch and feature dimensions drops below threshold.
        
        Args:
            z: Latent tensor, Shape: [B, N, D]
        """
        # Compute variance across batch and token space per channel
        # z: [B, N, D]
        # variance per feature channel:
        var = torch.var(z, dim=(0, 1)) # [D]
        mean_var = var.mean().item()
        
        return mean_var < self.var_threshold
        
    def check_hypersphere_pinning(self, z_t: torch.Tensor, z_prev: torch.Tensor) -> bool:
        """
        Checks if latent vectors are pinned to the same location across time steps
        (zero motion, constant output).
        
        Args:
            z_t: Current latent, Shape: [B, N, D]
            z_prev: Previous latent, Shape: [B, N, D]
        """
        # Flatten to compute average cosine similarity
        flat_zt = z_t.reshape(-1, z_t.shape[-1])
        flat_zprev = z_prev.reshape(-1, z_prev.shape[-1])
        
        norm_zt = F.normalize(flat_zt, p=2, dim=-1)
        norm_zprev = F.normalize(flat_zprev, p=2, dim=-1)
        
        cos_sim = torch.sum(norm_zt * norm_zprev, dim=-1) # [B*N]
        avg_cos_sim = cos_sim.mean().item()
        
        return avg_cos_sim >= self.pinning_threshold
        
    def verify_system_health(self, z_t: torch.Tensor, z_prev: torch.Tensor = None) -> tuple:
        """
        Performs all system checks.
        
        Args:
            z_t: Current latent, Shape: [B, N, D]
            z_prev: Previous latent, Shape: [B, N, D]
            
        Returns:
            healthy: Boolean indicating if system is fully operational.
            reason: String message detailing cause of any health breach.
        """
        # 1. NaN check
        if self.check_nan_inf(z_t):
            return False, "NaN/Infinity detected in latent representations"
            
        # 2. Variance collapse check
        if self.check_variance_collapse(z_t):
            return False, f"Variance Collapse detected (Var < {self.var_threshold})"
            
        # 3. Pinning check
        if z_prev is not None:
            if self.check_hypersphere_pinning(z_t, z_prev):
                return False, f"Hypersphere Pinning detected (Constant output similarity > {self.pinning_threshold})"
                
        return True, "SYSTEM_HEALTHY"
        
    def get_minimal_risk_action(self, current_action: torch.Tensor) -> torch.Tensor:
        """
        Bypasses the current action plans and returns the Minimal Risk Condition (MRC) action.
        For robotics, this typically corresponds to a zero-velocity, brake, or safe return home command.
        
        Args:
            current_action: Planned action tensor, Shape: [B, A]
            
        Returns:
            mrc_action: Zeroed-out/braking action tensor, Shape: [B, A]
        """
        # Deterministic MRC policy: return all zeros (direct braking command)
        # This is proven to be Lyapunov-stable (Theorem 5.2).
        return torch.zeros_like(current_action)

    def simulate_mrc_step(self, x_state: torch.Tensor, dt: float = 0.1) -> tuple:
        """
        Simulates one time-step under the Minimal Risk Condition (MRC) braking controller,
        and computes the Lyapunov function value and its derivative to verify stability.
        
        Args:
            x_state: Current physical state tensor, Shape: [B, 2]
            dt: Time step size
            
        Returns:
            next_state: Next physical state tensor, Shape: [B, 2]
            V_val: Lyapunov function value, Shape: [B]
            V_dot: Lyapunov derivative, Shape: [B]
        """
        # Linear control law: u = -K_gain * x
        u = -F.linear(x_state, self.K_gain) # [B, 1]
        
        # State derivatives: \dot{x} = A x + B u
        dx = F.linear(x_state, self.A) + F.linear(u, self.B) # [B, 2]
        
        # Euler integration
        next_state = x_state + dx * dt
        
        # Lyapunov function: V(x) = 0.5 * x^T P x
        V_val = 0.5 * torch.sum(x_state * F.linear(x_state, self.P), dim=-1) # [B]
        
        # Lyapunov derivative: \dot{V} = -0.5 * x^T Q x
        # where Q = - (A_cl^T P + P A_cl) = [[1.0, -0.5], [-0.5, 1.5]]
        Q = torch.tensor([[1.0, -0.5], [-0.5, 1.5]], device=x_state.device)
        V_dot = -0.5 * torch.sum(x_state * F.linear(x_state, Q), dim=-1) # [B]
        
        return next_state, V_val, V_dot
