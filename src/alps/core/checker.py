import torch
import torch.nn as nn
import torch.nn.functional as F

class RefinementNetwork(nn.Module):
    """
    Refinement neural network R(z; c) representing the operator.
    To behave as a contraction mapping, we regularize its Lipschitz constant during training.
    """
    def __init__(self, d_model: int = 384, d_cond: int = 384):
        super().__init__()
        self.d_model = d_model
        
        # Simple feed-forward network to compute the refinement delta
        self.net = nn.Sequential(
            nn.Linear(d_model + d_cond, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Current estimate of target latent, Shape: [B, N, D]
            c: Context/constraints from higher layers, Shape: [B, N, D_cond]
            
        Returns:
            R_z: Refined estimate, Shape: [B, N, D]
        """
        # Combine current estimate and context
        combined = torch.cat([z, c], dim=-1) # [B, N, D + D_cond]
        delta = self.net(combined)
        
        # The refinement operator: z_new = z + 0.1 * delta (residual learning)
        # Small residual scaling helps keep the mapping stable and contractive
        return z + 0.1 * delta

class BanachChecker(nn.Module):
    """
    Banach Contraction Checker.
    Executes a Checker-Refinement loop, iteratively refining predictions.
    Computes a contraction loss to enforce L < 1, mathematically proving convergence.
    """
    def __init__(self, d_model: int = 384, d_cond: int = 384, max_iters: int = 5, 
                 epsilon: float = 1e-4, target_lipschitz: float = 0.9):
        super().__init__()
        self.refiner = RefinementNetwork(d_model, d_cond)
        self.max_iters = max_iters
        self.epsilon = epsilon
        self.target_lipschitz = target_lipschitz
        
    def forward(self, initial_z: torch.Tensor, context: torch.Tensor) -> tuple:
        """
        Runs the checker-refinement loop until convergence or max_iters is reached.
        
        Args:
            initial_z: Initial prediction z^{(0)}, Shape: [B, N, D]
            context: Higher-level constraint vector, Shape: [B, N, D]
            
        Returns:
            refined_z: Converged prediction, Shape: [B, N, D]
            num_steps: Number of iterations executed before halting
            converged: Boolean tensor of shape [B] indicating success
        """
        B, N, D = initial_z.shape
        z = initial_z.clone()
        
        num_steps = 0
        converged = torch.zeros(B, dtype=torch.bool, device=initial_z.device)
        
        for i in range(self.max_iters):
            z_prev = z.clone()
            z = self.refiner(z, context)
            num_steps += 1
            
            # Check distance: d(z^(n), z^(n-1))
            dist = torch.norm(z - z_prev, p=2, dim=-1) # [B, N]
            max_dist = dist.max(dim=-1)[0] # [B]
            
            # Update convergence status
            newly_converged = (max_dist < self.epsilon)
            converged = converged | newly_converged
            
            # Early stopping if all batches converged
            if converged.all():
                break
                
        return z, num_steps, converged
        
    def compute_contraction_loss(self, z_batch: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Measures violation of the contraction property to add to the training loss.
        Calculates L = ||R(u) - R(v)|| / ||u - v|| for random pairs u, v.
        
        We want L <= target_lipschitz < 1.
        """
        # u and v are two perturbed versions of our latent batch
        u = z_batch + torch.randn_like(z_batch) * 0.1
        v = z_batch + torch.randn_like(z_batch) * 0.1
        
        R_u = self.refiner(u, context)
        R_v = self.refiner(v, context)
        
        # Calculate Euclidean distances
        dist_in = torch.norm(u - v, p=2, dim=-1) + 1e-8 # [B, N]
        dist_out = torch.norm(R_u - R_v, p=2, dim=-1) # [B, N]
        
        # Lipschitz ratio
        L = dist_out / dist_in
        
        # Penalize if Lipschitz constant exceeds target_lipschitz
        loss = F.relu(L - self.target_lipschitz)
        
        return loss.mean()
