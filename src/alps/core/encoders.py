import torch
import torch.nn as nn
import torch.nn.functional as F

class TubePatchEmbedding(nn.Module):
    """
    3D Spatiotemporal Patch Embedding for Video Inputs.
    Converts a video tensor of shape [B, C, T, H, W] into patch embeddings [B, N, D].
    """
    def __init__(self, in_channels: int = 3, d_model: int = 384, 
                 patch_size: tuple = (2, 16, 16)):
        super().__init__()
        self.patch_size = patch_size
        # Using 3D convolution to project 3D patches to embedding dimensions
        self.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        x = self.proj(x) # [B, D, T_out, H_out, W_out]
        x = x.flatten(2) # [B, D, N] where N = T_out * H_out * W_out
        x = x.transpose(1, 2) # [B, N, D]
        return x

class VisionEncoder(nn.Module):
    """
    3D Vision Transformer (ViT) Encoder for Video Latent Representation.
    Designed for ALPS-4B (22M parameters config).
    """
    def __init__(self, in_channels: int = 3, d_model: int = 384, depth: int = 8, 
                 num_heads: int = 6, mlp_ratio: float = 4.0, dropout: float = 0.1,
                 patch_size: tuple = (2, 16, 16), max_patches: int = 2048):
        super().__init__()
        self.d_model = d_model
        
        # 1. Spatiotemporal Tube Patch Embedding
        self.patch_embed = TubePatchEmbedding(
            in_channels=in_channels,
            d_model=d_model,
            patch_size=patch_size
        )
        
        # 2. Position Embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 3. Transformer Encoder Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 4. Final Norm
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask_indices: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Video tensor of shape [B, C, T, H, W]
            mask_indices: Optional boolean mask of shape [B, N] where True indicates masking.
                          If provided, masked patches are skipped during encoding (highly efficient).
                          
        Returns:
            Latent representations of shape [B, N_unmasked, D] or [B, N, D]
        """
        # Embed patches
        tokens = self.patch_embed(x) # [B, N, D]
        B, N, D = tokens.shape
        
        # Add position embeddings
        tokens = tokens + self.pos_embed[:, :N, :]
        
        if mask_indices is not None:
            # Perform asymmetric spatiotemporal masking (skipping masked patches to save compute)
            # mask_indices: [B, N] (True where patch is kept, False where patch is masked)
            # Note: active patches are selected.
            unmasked_tokens = []
            for b in range(B):
                active_t = tokens[b, mask_indices[b]] # [N_active, D]
                unmasked_tokens.append(active_t)
                
            # If active lengths differ, pad them or handle uniformly.
            # For JEPA masking, all batches typically mask the exact same indices per step,
            # or we can stack them if they have identical active lengths.
            tokens = torch.stack(unmasked_tokens, dim=0) # [B, N_active, D]
            
        # Transformer forward pass
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        
        return tokens
