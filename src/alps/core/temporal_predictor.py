"""
Causal temporal predictor (LeWM-style multi-frame history).

The original LeWorldModel predictor "takes as input a history of N frame
representations and predicts the next frame representation auto-regressively with
temporal causal masking" (sub-trajectories of 4 frames). Our previous predictor
used a single frame (z_t, a_t)->z_{t+1}. This module restores the multi-frame
history and — crucially — generalizes it to ALL three ALPS scales:

    operative : M = N spatial tokens per frame, horizon 1   (fast dynamics)
    tactical  : M = 1 pooled token per frame, horizon K_tac (mid abstraction)
    strategic : M = 1 pooled token per frame, horizon K_str (slow abstraction)

A single block-causal transformer handles any (K frames x M tokens): frame i
attends to all tokens of frames <= i. Per-frame conditioning (action for the
operative; strategic concept for the tactical; etc.) is injected additively.
Trained teacher-forced over a window, the output at frame i predicts the target
`horizon` steps ahead (i+horizon), so one forward pass yields many supervised
predictions.
"""

from __future__ import annotations
import torch
import torch.nn as nn


def block_causal_mask(K: int, M: int, device) -> torch.Tensor:
    """[K*M, K*M] bool mask; True = NOT allowed (key is in a strictly future frame)."""
    frame = torch.arange(K * M, device=device) // M     # frame index per position
    return frame[None, :] > frame[:, None]              # key frame > query frame -> mask


class CausalTemporalPredictor(nn.Module):
    def __init__(self, d_model: int, d_cond: int, depth: int = 4, num_heads: int = 6,
                 mlp_ratio: float = 4.0, max_frames: int = 12, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            for h in range(num_heads, 0, -1):
                if d_model % h == 0:
                    num_heads = h
                    break
        self.frame_emb = nn.Parameter(torch.randn(1, max_frames, 1, d_model) * 0.02)
        self.cond_proj = nn.Sequential(nn.Linear(d_cond, d_model), nn.GELU(),
                                       nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Linear(d_model, d_model)

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """z [B,K,M,D], cond [B,K,d_cond] -> pred [B,K,M,D].

        pred[:, i] is conditioned on frames 0..i and on cond[:, i]; with the
        teacher-forced loss it is trained to equal the latent `horizon` steps
        after frame i. At inference the last position pred[:, K-1] is the
        prediction of the next (beyond-window) frame.
        """
        B, K, M, D = z.shape
        cemb = self.cond_proj(cond).unsqueeze(2)        # [B,K,1,D]
        x = (z + self.frame_emb[:, :K] + cemb).reshape(B, K * M, D)
        x = self.transformer(x, mask=block_causal_mask(K, M, z.device))
        return self.head(x).reshape(B, K, M, D)

    @torch.no_grad()
    def predict_next(self, z_hist: torch.Tensor, cond_hist: torch.Tensor) -> torch.Tensor:
        """Given a history window [B,K,M,D] + cond [B,K,d_cond], return the next
        frame prediction [B,M,D] (the last causal position)."""
        return self.forward(z_hist, cond_hist)[:, -1]
