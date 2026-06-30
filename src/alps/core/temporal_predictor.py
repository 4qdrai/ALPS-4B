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
                 mlp_ratio: float = 4.0, max_frames: int = 12, dropout: float = 0.1,
                 residual: bool = False, film_cond: bool = False,
                 flow: bool = False, flow_steps: int = 8, flow_samples: int = 4):
        super().__init__()
        if d_model % num_heads != 0:
            for h in range(num_heads, 0, -1):
                if d_model % h == 0:
                    num_heads = h
                    break
        # residual/delta prediction: the head outputs the CHANGE from frame i and we add
        # back z[:, i], with the action re-injected just before the head. The static
        # background is carried by the skip for free, so all capacity goes to the
        # action-driven delta (the moving object) -- the part the uniform next-latent MSE
        # otherwise under-learns (boilerplate domination). See diagnose_control / BLOCK_ROOMS.
        self.residual = residual
        # per-layer (FiLM-lite) conditioning: re-inject the action before EVERY transformer
        # layer instead of once at the input. Injected once, the action is washed out through
        # a deep stack -> the predicted consequence keeps the right magnitude but the wrong
        # DIRECTION (calibrated dir_acc 0.41, action_spread ~= true step). Re-injecting per
        # layer keeps the action driving the prediction all the way through -> sharper
        # action->direction mapping. See the manifold-mismatch diagnosis.
        self.film_cond = film_cond
        self.frame_emb = nn.Parameter(torch.randn(1, max_frames, 1, d_model) * 0.02)
        self.cond_proj = nn.Sequential(nn.Linear(d_cond, d_model), nn.GELU(),
                                       nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Linear(d_model, d_model)
        # FLOW-MATCHING predictor head (--flow-pred). The transformer encodes the per-token
        # CONTEXT (history + action); the flow head is a conditional velocity field that
        # transports noise -> a SHARP SAMPLE of the next latent, instead of the MSE head's
        # blurry conditional mean. For deterministic dynamics the sample collapses to the
        # on-manifold point, so each token commits to block-or-floor rather than averaging
        # -> fixes the off-manifold blur that made the imagined readout direction-random.
        self.flow = flow
        self.flow_steps = flow_steps        # Euler integration steps at inference
        self.flow_samples = flow_samples    # samples averaged at inference (variance ↓)
        if flow:
            self.time_mlp = nn.Sequential(nn.Linear(1, d_model), nn.GELU(),
                                          nn.Linear(d_model, d_model))
            h = 2 * d_model
            self.flow_head = nn.Sequential(nn.Linear(3 * d_model, h), nn.GELU(),
                                           nn.Linear(h, h), nn.GELU(),
                                           nn.Linear(h, d_model))

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """z [B,K,M,D], cond [B,K,d_cond] -> pred [B,K,M,D].

        pred[:, i] is conditioned on frames 0..i and on cond[:, i]; with the
        teacher-forced loss it is trained to equal the latent `horizon` steps
        after frame i. At inference the last position pred[:, K-1] is the
        prediction of the next (beyond-window) frame.
        """
        B, K, M, D = z.shape
        x, cemb = self.encode_context(z, cond, return_cemb=True)   # [B,K,M,D]
        if self.residual:
            # re-inject the per-frame conditioning right before the head so the action
            # directly shapes the delta, then add the skip: pred[:, i] = z[:, i] + delta.
            cfull = cemb.expand(-1, -1, M, -1)
            delta = self.head(x + cfull)
            return z + delta
        return self.head(x)

    def encode_context(self, z: torch.Tensor, cond: torch.Tensor, return_cemb: bool = False):
        """[B,K,M,D],[B,K,d_cond] -> per-token context [B,K,M,D] (the causal transformer
        output, before any head). Shared by the MSE head, the residual head, and the flow
        velocity field. cond[:, i] conditions frames 0..i (block-causal)."""
        B, K, M, D = z.shape
        cemb = self.cond_proj(cond).unsqueeze(2)        # [B,K,1,D]
        mask = block_causal_mask(K, M, z.device)
        x = (z + self.frame_emb[:, :K] + cemb).reshape(B, K * M, D)
        if self.film_cond:
            # re-inject the conditioning before every layer so the action keeps driving the
            # prediction through the full depth (not washed out after the input).
            cfull = cemb.expand(-1, -1, M, -1).reshape(B, K * M, D)
            for layer in self.transformer.layers:
                x = layer(x + cfull, src_mask=mask)
        else:
            x = self.transformer(x, mask=mask)
        x = x.reshape(B, K, M, D)
        return (x, cemb) if return_cemb else x

    # ---- flow-matching head (conditional rectified flow over the next latent) ----
    def velocity(self, z_tau: torch.Tensor, t, ctx: torch.Tensor) -> torch.Tensor:
        """Conditional velocity field v(z_tau, t | ctx). z_tau, ctx [...,D]; t scalar or
        tensor broadcastable to [...,1]."""
        if not torch.is_tensor(t):
            t = torch.full_like(z_tau[..., :1], float(t))
        t_emb = self.time_mlp(t)
        return self.flow_head(torch.cat([z_tau, ctx, t_emb], dim=-1))

    def flow_loss(self, ctx: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        """Conditional flow-matching loss: learn to transport noise z0~N(0,I) to the target
        z1 conditioned on ctx. Target velocity is z1 - z0 along the straight path
        z_tau = (1-tau) z0 + tau z1 (rectified flow). ctx, z1 [...,D]."""
        z0 = torch.randn_like(z1)
        tau = torch.rand_like(z1[..., :1])
        z_tau = (1.0 - tau) * z0 + tau * z1
        return torch.nn.functional.mse_loss(self.velocity(z_tau, tau, ctx), z1 - z0)

    @torch.no_grad()
    def flow_sample(self, ctx: torch.Tensor, n_steps: int = None, n_samples: int = None) -> torch.Tensor:
        """Sample the next latent by integrating dz/dtau = v(z, tau | ctx) from noise over
        [0,1] (forward Euler). Averages n_samples draws (variance reduction; ~deterministic
        dynamics => low variance). ctx [...,D] -> [...,D]."""
        n_steps = self.flow_steps if n_steps is None else n_steps
        n_samples = self.flow_samples if n_samples is None else n_samples
        acc = None
        dt = 1.0 / n_steps
        for _ in range(max(1, n_samples)):
            z = torch.randn_like(ctx)
            for s in range(n_steps):
                z = z + dt * self.velocity(z, s * dt, ctx)
            acc = z if acc is None else acc + z
        return acc / max(1, n_samples)

    @torch.no_grad()
    def predict_next(self, z_hist: torch.Tensor, cond_hist: torch.Tensor) -> torch.Tensor:
        """Given a history window [B,K,M,D] + cond [B,K,d_cond], return the next
        frame prediction [B,M,D] (the last causal position)."""
        if self.flow:
            ctx = self.encode_context(z_hist, cond_hist)[:, -1]    # [B,M,D] context at last frame
            return self.flow_sample(ctx)
        return self.forward(z_hist, cond_hist)[:, -1]
