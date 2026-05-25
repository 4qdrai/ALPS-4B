import torch
import torch.nn as nn
import torch.nn.functional as F

class SIGReg(nn.Module):
    """
    SIGReg: Sketched Isotropic Gaussian Regularization
    
    This module implements collapse prevention in Joint-Embedding Predictive Architectures (JEPAs)
    without momentum encoders or stop-gradients, as described in LeWorldModel (arXiv:2603.19312) 
    and LeJEPA.
    
    SIGReg projects high-dimensional latent vectors onto random 1D directions and applies
    the analytical Epps-Pulley normality test statistic. Enforcing normality across 
    random 1D projections guarantees that the joint latent distribution matches an isotropic
    multivariate Gaussian, preventing both representation and dimensional collapse.
    
    Features:
    1. Full SIGReg (Epps-Pulley Characteristic Function goodness-of-fit).
    2. Weak-SIGReg (Covariance-based sketching, computationally cheaper).
    """
    def __init__(self, d_model: int, num_slices: int = 128, beta: float = 1.0, weak_only: bool = False):
        super().__init__()
        self.d_model = d_model
        self.num_slices = num_slices
        self.beta = beta
        self.weak_only = weak_only
        
        # We store the random projection matrix as a registered buffer.
        # It gets re-initialized periodically, or is kept static to stabilize training.
        self.register_buffer("projection_matrix", torch.randn(d_model, num_slices))
        self.reset_projections()
        
    def reset_projections(self):
        """Generates new random orthornormal or isotropic projection vectors on the unit sphere."""
        device = self.projection_matrix.device
        # Generate random directions from an isotropic Gaussian
        W = torch.randn(self.d_model, self.num_slices, device=device)
        # Normalize each column to be a unit vector on the hypersphere
        norms = torch.norm(W, dim=0, keepdim=True)
        self.projection_matrix.copy_(W / (norms + 1e-8))
        
    def epps_pulley_statistic(self, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the analytical Epps-Pulley test statistic for each slice.
        
        Args:
            y: Standardized 1D projections, Shape: [N, num_slices]
               Each column represents a batch of projections on a specific random unit vector.
               We expect each column to follow a standard normal N(0, 1).
               
        Returns:
            The mean Epps-Pulley statistic across all slices.
        """
        N = y.shape[0]
        if N <= 1:
            return torch.tensor(0.0, device=y.device, requires_grad=True)
            
        beta2 = self.beta ** 2
        
        # 1. Double summation term: sum_{j=1}^N sum_{k=1}^N exp(-beta^2 / 2 * (y_j - y_k)^2)
        # We vectorise this over slices.
        # y: [N, S] where S = num_slices
        # We can expand y to [N, 1, S] and [1, N, S] to compute pairwise differences.
        y_expanded1 = y.unsqueeze(1) # [N, 1, S]
        y_expanded2 = y.unsqueeze(0) # [1, N, S]
        diff2 = (y_expanded1 - y_expanded2) ** 2 # [N, N, S]
        term1 = torch.exp(-0.5 * beta2 * diff2).sum(dim=(0, 1)) / N # [S]
        
        # 2. Single summation term: -2 * (1 + beta^2)^(-1/2) * sum_{j=1}^N exp(-beta^2 * y_j^2 / (2 * (1 + beta^2)))
        coeff2 = -2.0 / math_sqrt(1.0 + beta2)
        exponent2 = -0.5 * beta2 * (y ** 2) / (1.0 + beta2)
        term2 = coeff2 * torch.exp(exponent2).sum(dim=0) # [S]
        
        # 3. Constant term: N * (1 + 2 * beta^2)^(-1/2)
        term3 = N / math_sqrt(1.0 + 2.0 * beta2)
        
        # Total statistic for each slice
        T = (term1 + term2 + term3)
        
        # Return the average across all projection slices
        return T.mean()

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
            
        # Dynamically refresh projections occasionally in eval or training to ensure full coverage
        # In practice, static random matrices during a step are sufficient.
        
        # Project latents onto the random slices
        # z: [N, D], projection_matrix: [D, S] -> y: [N, S]
        y = torch.matmul(z, self.projection_matrix)
        
        if self.weak_only or N > 1024:
            # Weak-SIGReg: Frobenius norm of covariance of projected features minus Identity
            # Cov(y) = 1/(N-1) * (y - mean_y)^T * (y - mean_y)
            mean_y = y.mean(dim=0, keepdim=True)
            y_centered = y - mean_y
            cov = torch.matmul(y_centered.t(), y_centered) / (N - 1)
            # We want Cov(y) to be close to the identity matrix
            identity = torch.eye(cov.shape[0], device=cov.device)
            loss = torch.norm(cov - identity, p="fro") ** 2 / cov.shape[0]
            return loss
            
        # Standardize each slice to zero mean and unit variance (empirical standardization)
        # so that it can be compared to N(0, 1) using Epps-Pulley.
        mean = y.mean(dim=0, keepdim=True)
        std = y.std(dim=0, keepdim=True, unbiased=True) + 1e-6
        y_standardized = (y - mean) / std
        
        # Compute Epps-Pulley statistic
        return self.epps_pulley_statistic(y_standardized)

def math_sqrt(x: float) -> float:
    import math
    return math.sqrt(x)
