import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN)
    Modulates standard layer normalization parameters (scale and shift) using an external conditioning signal (e.g. actions).
    """
    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.fc = nn.Linear(d_cond, d_model * 2)
        # Initialize scale to zero and bias to zero so it initially behaves like standard LayerNorm
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, Shape: [B, N, D]
            cond: Conditioning tensor (e.g., action), Shape: [B, D_cond] or [B, N, D_cond]
        """
        # Normalize x
        x_norm = self.norm(x)
        
        # Project conditioning vector to scale (gamma) and shift (beta)
        if cond.dim() == 2:
            # Broadcast across patch sequence N
            scale_shift = self.fc(cond).unsqueeze(1) # [B, 1, 2 * D]
        else:
            scale_shift = self.fc(cond) # [B, N, 2 * D]
            
        scale, shift = torch.chunk(scale_shift, 2, dim=-1) # [B, N, D], [B, N, D]
        
        # Apply modulation: x_norm * (1 + scale) + shift
        return x_norm * (1.0 + scale) + shift

class AdaLNTransformerBlock(nn.Module):
    """
    Transformer Block with Adaptive Layer Normalization (AdaLN) for conditioning.
    """
    def __init__(self, d_model: int, nhead: int, d_cond: int, dim_feedforward: int = 1536, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln1 = AdaLN(d_model, d_cond)
        self.ln2 = AdaLN(d_model, d_cond)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input token sequence, Shape: [B, N, D]
            cond: Conditioning tensor, Shape: [B, D_cond]
        """
        # Self-Attention Branch
        x_norm = self.ln1(x, cond)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP Branch
        x_norm2 = self.ln2(x, cond)
        mlp_out = self.mlp(x_norm2)
        x = x + mlp_out
        
        return x

class MultiScalePredictor(nn.Module):
    """
    Spatiotemporal Predictor network for JEPA layers.
    Predicts the next latent representation z_{t+1} given current latent z_t and an action/guidance.
    Can be configured for Operative, Tactical, or Strategic timescales.
    """
    def __init__(self, d_model: int = 384, d_cond: int = 64, depth: int = 6, 
                 num_heads: int = 6, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Action conditioning projection
        self.cond_proj = nn.Sequential(
            nn.Linear(d_cond, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Stack of AdaLN Transformer Blocks
        self.blocks = nn.ModuleList([
            AdaLNTransformerBlock(
                d_model=d_model,
                nhead=num_heads,
                d_cond=d_model,
                dim_feedforward=int(d_model * mlp_ratio),
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        # Final prediction head (maps back to target latent dimension if needed, otherwise identity mapping)
        self.head = nn.Linear(d_model, d_model)
        
    def forward(self, z: torch.Tensor, cond_signal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent tokens at step t, Shape: [B, N, D]
            cond_signal: Gating signal (e.g. action vector or top-down guidance token), Shape: [B, D_cond]
            
        Returns:
            Predicted latent tokens for step t+1, Shape: [B, N, D]
        """
        # Embed conditioning signal
        cond_emb = self.cond_proj(cond_signal) # [B, D]
        
        # Forward pass through AdaLN blocks
        h = z
        for block in self.blocks:
            h = block(h, cond_emb)
            
        # Linear projection
        out = self.head(h)
        return out
