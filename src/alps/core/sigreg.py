import torch
import torch.nn as nn
import torch.nn.functional as F

class SIGReg(nn.Module):
    """
    SIGReg: Sketched Isotropic Gaussian Regularization
    
    This module implements collapse prevention in Joint-Embedding Predictive Architectures (JEPAs)
    without momentum encoders or stop-gradients, as described in LeWorldModel (arXiv:2603.19312) 
    and LeJEPA.
    
    This implementation aligns exactly with the official knots-based Epps-Pulley characteristic function
    normality check from the LeWorldModel codebase. It runs in O(N) memory (avoiding O(N^2) pairwise 
    VRAM crashes) and operates directly on raw projections without standardizing the batch variance,
    thus closing the scaling loophole and robustly preventing representation collapse.
    """
    def __init__(self, d_model: int, num_slices: int = 1024, knots: int = 17, weak_only: bool = False, beta: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.num_slices = num_slices
        self.weak_only = weak_only
        self.beta = beta
        
        # Knots setup for Epps-Pulley characteristic function check
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)
        
        # Store random projections matrix
        self.register_buffer("projection_matrix", torch.randn(d_model, num_slices))
        self.reset_projections()
        
    def reset_projections(self):
        """Generates new random projection vectors on the unit sphere."""
        device = self.projection_matrix.device
        W = torch.randn(self.d_model, self.num_slices, device=device)
        norms = torch.norm(W, dim=0, keepdim=True)
        self.projection_matrix.copy_(W / (norms + 1e-8))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes SIGReg collapse prevention loss.
        
        Args:
            z: Latent embeddings, Shape: [N, D] or [B, T, D]
            
        Returns:
            Scalar SIGReg loss term.
        """
        # Collapse dimensions to [N, D] where N = B * T
        if z.dim() == 3:
            N, T, D = z.shape
            z = z.reshape(N * T, D)
        elif z.dim() > 3 or z.dim() < 2:
            raise ValueError(f"SIGReg expects 2D or 3D tensor, got shape {z.shape}")
            
        N, D = z.shape
        if N <= 1:
            return torch.tensor(0.0, device=z.device, requires_grad=True)
            
        if self.weak_only:
            # Weak-SIGReg: Frobenius norm of covariance of native features minus Identity.
            # Cov(z) = 1/(N-1) * (z - mean_z)^T * (z - mean_z)
            mean_z = z.mean(dim=0, keepdim=True)
            z_centered = z - mean_z
            cov = torch.matmul(z_centered.t(), z_centered) / (N - 1)
            identity = torch.eye(cov.shape[0], device=cov.device)
            loss = torch.norm(cov - identity, p="fro") ** 2 / cov.shape[0]
            return loss

        # Standard SIGReg (from the LeWorldModel paper)
        # 1. Project latents onto random unit-norm directions
        # z: [N, D], projection_matrix: [D, S] -> y: [N, S]
        y = torch.matmul(z, self.projection_matrix)
        
        # 2. Compute Epps-Pulley characteristic function difference linearly (O(N) memory)
        # x_t: [N, S, K] where K = knots
        x_t = y.unsqueeze(-1) * self.t
        
        # Empirical characteristic function: mean(cos(x*t)) and mean(sin(x*t)) over the batch (dim=0)
        cos_term = x_t.cos().mean(dim=0) - self.phi  # [S, K]
        sin_term = x_t.sin().mean(dim=0)             # [S, K]
        
        err = cos_term.square() + sin_term.square()   # [S, K]
        
        # Epps-Pulley statistic scales linearly with the batch size N
        statistic = (err @ self.weights) * N         # [S]
        
        return statistic.mean()
