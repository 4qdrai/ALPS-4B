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
    
    When spatiotemporal tube masking is applied, only the unmasked (active) 10% of
    patches are processed through the expensive Transformer layers. The output is then
    scattered back into a full-length sequence using a learnable [MASK] token, ensuring
    the encoder ALWAYS returns shape [B, N, D] regardless of masking. This follows the
    MAE / I-JEPA design pattern for asymmetric encoder-predictor architectures.
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
        
        # 3. Learnable [MASK] token for reconstructing full sequence after sparse encoding
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # 4. Transformer Encoder Blocks
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
        
        # 5. Final Norm
        self.norm = nn.LayerNorm(d_model)
        
        # 6. Projection Head (MLP + LayerNorm) — Critical for SIGReg stability (LeWM §3.2)
        # Projects latent features through a non-linear bottleneck before SIGReg is applied.
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        
    def forward(self, x: torch.Tensor, mask_indices: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Video tensor of shape [B, C, T, H, W]
            mask_indices: Optional boolean mask of shape [B, N] where True indicates patches
                          to KEEP (unmasked), and False indicates patches to MASK.
                          If provided, only unmasked patches are processed through the Transformer
                          (massive compute savings), then scattered back to full length.
                          
        Returns:
            Latent representations of shape [B, N, D] (always full sequence length)
        """
        # Embed patches
        tokens = self.patch_embed(x) # [B, N, D]
        B, N, D = tokens.shape
        
        # Add position embeddings
        tokens = tokens + self.pos_embed[:, :N, :]
        
        if mask_indices is not None:
            # --- Asymmetric Sparse Encoding (10x compute savings at 90% mask ratio) ---
            # Step 1: Extract only unmasked (active) tokens for transformer processing
            unmasked_tokens = []
            for b in range(B):
                active_t = tokens[b, mask_indices[b]] # [N_active, D]
                unmasked_tokens.append(active_t)
                
            active_tokens = torch.stack(unmasked_tokens, dim=0) # [B, N_active, D]
            
            # Step 2: Transformer forward on sparse active tokens only (O(N_active^2) not O(N^2))
            active_tokens = self.transformer(active_tokens)
            active_tokens = self.norm(active_tokens)
            active_tokens = self.projection_head(active_tokens)
            
            # Step 3: Scatter back into full-length sequence using learnable [MASK] tokens
            # Fill with mask_token + positional embeddings for masked positions
            full_tokens = self.mask_token.expand(B, N, -1).clone() + self.pos_embed[:, :N, :]
            for b in range(B):
                full_tokens[b, mask_indices[b]] = active_tokens[b]
                
            return full_tokens
        else:
            # Dense forward pass (no masking)
            tokens = self.transformer(tokens)
            tokens = self.norm(tokens)
            return self.projection_head(tokens)
