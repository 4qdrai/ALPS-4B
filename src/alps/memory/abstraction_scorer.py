import torch
import torch.nn as nn
import torch.nn.functional as F

class AbstractionScorer(nn.Module):
    """
    Abstraction Scorer.
    
    Provides formal mathematical metrics to analyze the abstraction level of any
    latent representation z, determining its routing to the appropriate memory tier.
    
    Metrics:
    1. Temporal Invariance (I_temp):
       I_temp = 1.0 - tanh(|| z_t - z_{t-1} ||_2).
       High value indicates a slow-varying, strategic concept.
       
    2. Dimensional Compression (C_dim):
       Measured via the Effective Rank (EffRank) of the covariance matrix of z.
       EffRank = exp(- sum p_i * ln p_i), where p_i are normalized eigenvalues.
       High compression (low EffRank) represents clean, low-dimensional semantic coordinates.
    """
    def __init__(self, d_model: int = 384):
        super().__init__()
        self.d_model = d_model
        
    def compute_temporal_invariance(self, z_t: torch.Tensor, z_prev: torch.Tensor) -> float:
        """
        Computes the temporal invariance score.
        
        Args:
            z_t: Current latent, Shape: [B, N, D]
            z_prev: Previous latent, Shape: [B, N, D]
            
        Returns:
            invariance: Scalar float in [0, 1] where 1 indicates absolute temporal stability.
        """
        # Temporal derivative approximation
        diff = z_t - z_prev
        norm_diff = torch.norm(diff, p=2, dim=-1).mean().item()
        
        # Normalize via tanh to [0, 1] range
        # Slow-varying signals (small norm_diff) yield scores close to 1
        return 1.0 - float(torch.tanh(torch.tensor(norm_diff)).item())
        
    def compute_dimensional_compression(self, z: torch.Tensor) -> float:
        """
        Computes the effective rank of the latent covariance matrix.
        
        Args:
            z: Latents batch, Shape: [B, N, D]
            
        Returns:
            eff_rank: Effective rank value in [1, D_model].
                      Lower values indicate high dimensional compression (highly structured).
        """
        B, N, D = z.shape
        flat_z = z.reshape(B * N, D)
        
        if flat_z.shape[0] <= 1:
            return 1.0
            
        # Center z
        centered_z = flat_z - flat_z.mean(dim=0, keepdim=True)
        
        # SVD of centered representations to extract singular values (proportional to sqrt of eigenvalues)
        try:
            # We use SVDecon or PyTorch SVD
            _, S, _ = torch.linalg.svd(centered_z, full_matrices=False)
        except RuntimeError:
            # SVD convergence failure fallback
            return float(D)
            
        # Eigenvalues approximation
        eigenvalues = S ** 2
        sum_eigen = eigenvalues.sum() + 1e-8
        
        # Probability distribution over eigenvalues
        p = eigenvalues / sum_eigen
        p = torch.clamp(p, min=1e-8)
        
        # Shannon Entropy of the spectrum
        entropy = -torch.sum(p * torch.log(p))
        
        # Effective Rank
        eff_rank = torch.exp(entropy).item()
        
        return float(eff_rank)
        
    def classify_abstraction_tier(self, z_t: torch.Tensor, z_prev: torch.Tensor = None) -> str:
        """
        Classifies which hierarchical tier a representation belongs to.
        
        Classification heuristics:
        - Low effective rank + high temporal invariance -> STRATEGIC
        - Medium effective rank + medium temporal invariance -> TACTICAL
        - High effective rank + low temporal invariance -> OPERATIVE
        """
        eff_rank = self.compute_dimensional_compression(z_t)
        
        # We normalize effective rank to a compression ratio in [0, 1]
        compression_ratio = 1.0 - (eff_rank / self.d_model)
        
        if z_prev is not None:
            invariance = self.compute_temporal_invariance(z_t, z_prev)
        else:
            invariance = 0.5 # default midpoint if no temporal context
            
        # Combine metrics into an abstraction score
        score = 0.5 * compression_ratio + 0.5 * invariance
        
        if score >= 0.7:
            return "STRATEGIC"
        elif score >= 0.4:
            return "TACTICAL"
        else:
            return "OPERATIVE"
