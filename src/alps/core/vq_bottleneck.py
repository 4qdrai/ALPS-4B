import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    Vector Quantization (VQ) Bottleneck Layer.
    Enforces discrete concept learning in the Strategic Layer by quantizing
    continuous latents onto a finite conceptual codebook.
    
    Includes Straight-Through Estimator (STE) for passing gradients back to the encoder,
    and a commitment loss term to bind the continuous representations.
    """
    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 384, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Initialize codebook embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
    def forward(self, inputs: torch.Tensor) -> tuple:
        """
        Args:
            inputs: Continuous embeddings, Shape: [B, N, D]
            
        Returns:
            quantized: Quantized discrete embeddings, Shape: [B, N, D]
            loss: Codebook commitment loss term (scalar)
            encoding_indices: Index of the closest codebook vector for each patch, Shape: [B, N]
        """
        # Flatten inputs to [B*N, D] for distance calculations
        flat_inputs = inputs.reshape(-1, self.embedding_dim)
        
        # Calculate distances: ||inputs||^2 + ||embeddings||^2 - 2 * inputs * embeddings
        # flat_inputs: [B*N, D], embeddings: [C, D]
        distances = (
            torch.sum(flat_inputs ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_inputs, self.embeddings.weight.t())
        ) # [B*N, num_embeddings]
        
        # Find closest codebook vectors
        encoding_indices = torch.argmin(distances, dim=1) # [B*N]
        
        # Retrieve quantized vectors
        quantized_flat = self.embeddings(encoding_indices) # [B*N, D]
        quantized = quantized_flat.reshape(inputs.shape) # [B, N, D]
        
        # Commitment and Codebook loss terms
        # 1. Commitment Loss: forces continuous representations to commit to codebook vectors
        loss_commitment = F.mse_loss(quantized.detach(), inputs)
        
        # 2. Codebook Loss: moves codebook vectors closer to continuous embeddings
        loss_codebook = F.mse_loss(quantized, inputs.detach())
        
        loss = loss_codebook + self.commitment_cost * loss_commitment
        
        # Straight-Through Estimator (STE):
        # We bypass quantization in the backward pass.
        # This keeps the gradient of 'inputs' active.
        quantized = inputs + (quantized - inputs).detach()
        
        # Reshape indices to [B, N]
        indices = encoding_indices.reshape(inputs.shape[0], inputs.shape[1])
        
        return quantized, loss, indices
