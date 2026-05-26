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
            
        # 3. Pinning check (only valid if shapes match across time steps)
        if z_prev is not None and z_t.shape == z_prev.shape:
            if self.check_hypersphere_pinning(z_t, z_prev):
                return False, f"Hypersphere Pinning detected (Constant output similarity > {self.pinning_threshold})"
                
        return True, "SYSTEM_HEALTHY"
        
    def get_minimal_risk_action(self, current_action: torch.Tensor, current_position: torch.Tensor = None) -> torch.Tensor:
        """
        Bypasses the current action plans and returns the Minimal Risk Condition (MRC) action.
        If current_position is provided, it steers the agent to the nearest safe haven.
        Otherwise, it returns a zeroed-out safe stop/braking command.
        
        Args:
            current_action: Planned action tensor, Shape: [B, A]
            current_position: Current 2D positions of the agent, Shape: [B, 2]
            
        Returns:
            mrc_action: Active homing action (one-hot) or safe braking (all zeros)
        """
        if current_position is None:
            # Deterministic MRC policy: return all zeros (direct braking command)
            return torch.zeros_like(current_action)
            
        B = current_position.shape[0]
        device = current_position.device
        
        # 1. Define Safe Havens in the Two Rooms environment:
        # Left Room Center = (2.5, 5.0), Right Room Center = (7.5, 5.0)
        safe_havens = torch.tensor([[2.5, 5.0], [7.5, 5.0]], device=device, dtype=torch.float32) # [2, 2]
        
        # 2. Compute distance from each batch position to both safe havens
        dist = torch.norm(current_position.unsqueeze(1) - safe_havens.unsqueeze(0), p=2, dim=-1) # [B, 2]
        
        # Find index of closest haven per batch element
        closest_haven_idx = torch.argmin(dist, dim=-1) # [B]
        
        # Select the target safe haven coordinates
        target_havens = safe_havens[closest_haven_idx] # [B, 2]
        
        # 3. Compute directional vector to the target
        d_pos = target_havens - current_position # [B, 2]
        dx = d_pos[:, 0]
        dy = d_pos[:, 1]
        
        # 4. Generate one-hot MRC steering action
        # Actions: 0=up (+y), 1=down (-y), 2=left (-x), 3=right (+x)
        # Action step size is 0.3. If we are within 0.3 distance, we safe halt (return zeros).
        mrc_action = torch.zeros_like(current_action) # [B, A]
        
        for b in range(B):
            dist_to_haven = dist[b, closest_haven_idx[b]].item()
            if dist_to_haven <= 0.3:
                # Already at safe haven, output zero (safe stop)
                continue
                
            dx_b = dx[b].item()
            dy_b = dy[b].item()
            
            if abs(dx_b) > abs(dy_b):
                # Move horizontally
                if dx_b > 0:
                    mrc_action[b, 3] = 1.0  # right
                else:
                    mrc_action[b, 2] = 1.0  # left
            else:
                # Move vertically
                if dy_b > 0:
                    mrc_action[b, 0] = 1.0  # up
                else:
                    mrc_action[b, 1] = 1.0  # down
                    
        return mrc_action

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
