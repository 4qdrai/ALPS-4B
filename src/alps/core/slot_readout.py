"""Object-centric SLOT readout — the size-invariant fix for the dense-grid dilution problem.

The grid-pool + ridge (and even soft-argmax) read a SMALL agent poorly: it is a tiny fraction
of dense tokens and blends with same-size distractors (key/target). Slot Attention (Locatello
et al. 2020) instead lets a small set of learned SLOTS compete to bind the encoder tokens, so
each object (agent, key, target) is captured by its OWN slot -- INDEPENDENT of pixel size. A
per-slot position head + a soft agent-selector then read the agent's position from its slot.

Two properties that matter here:
  1. SIZE-INVARIANT decode: the agent slot binds by FEATURE (colour/appearance), not coverage,
     so a 0.5-radius agent reads as sharply as a 1.7-radius one.
  2. Reads a DIFFUSE imagination: slot attention AGGREGATES the (distributed) agent activation
     of the predictor's off-manifold output into one slot -> recovers the imagined position even
     when the peak is blurry (where soft-argmax's peak-centroid reads flat).

Fully SELF-SUPERVISED as a frozen measuring probe: only the slots + tiny heads are fit on
proprioceptive positions; the encoder/predictors are untouched. This is the readout half of the
slot-structured four-brain (docs/SLOT_FOUR_BRAIN.md); the predictor half predicts slot dynamics.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class SlotAttention(nn.Module):
    def __init__(self, dim: int, num_slots: int = 4, iters: int = 3, hidden: int = None):
        super().__init__()
        self.num_slots, self.iters, self.scale = num_slots, iters, dim ** -0.5
        hidden = hidden or dim
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim) * 0.1)
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        self.to_q = nn.Linear(dim, dim); self.to_k = nn.Linear(dim, dim); self.to_v = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.norm_in = nn.LayerNorm(dim); self.norm_slots = nn.LayerNorm(dim); self.norm_mlp = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x [B,N,D] -> slots [B,K,D]
        B, N, D = x.shape
        x = self.norm_in(x); k = self.to_k(x); v = self.to_v(x)
        mu = self.slots_mu.expand(B, self.num_slots, -1)
        slots = mu + self.slots_logsigma.exp().expand(B, self.num_slots, -1) * torch.randn_like(mu)
        for _ in range(self.iters):
            q = self.to_q(self.norm_slots(slots))
            attn = torch.softmax((q @ k.transpose(1, 2)) * self.scale, dim=1)     # softmax over SLOTS (compete)
            attn = attn / (attn.sum(-1, keepdim=True) + 1e-8)                     # normalise over tokens
            updates = attn @ v                                                    # [B,K,D]
            slots = self.gru(updates.reshape(-1, D), slots.reshape(-1, D)).reshape(B, self.num_slots, D)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots


class SlotPositionReadout(nn.Module):
    """Slot attention -> per-slot position + agent-score -> soft-selected AGENT position [.., 2]."""
    def __init__(self, dim: int, num_slots: int = 4, iters: int = 3):
        super().__init__()
        self.slots = SlotAttention(dim, num_slots, iters)
        self.pos_head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 2))
        self.agent_head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # [.., N, D] -> [.., 2]
        shp = x.shape[:-2]
        s = self.slots(x.reshape(-1, x.shape[-2], x.shape[-1]))                   # [B,K,D]
        pos = self.pos_head(s)                                                    # [B,K,2]
        w = torch.softmax(self.agent_head(s).squeeze(-1), dim=-1)                 # [B,K] soft agent selector
        return (w.unsqueeze(-1) * pos).sum(1).reshape(*shp, 2)


@torch.no_grad()
def _grids(m, frames, idx, dev, bs=128):
    from alps.evaluation.validate_temporal import _gather_token_grids
    return _gather_token_grids(m, frames, idx, dev)


def fit_slot_decode(token_grids: torch.Tensor, Y: torch.Tensor, dev, num_slots=4, iters=3,
                    epochs=300, lr=3e-3, bs=256):
    """Fit a frozen SlotPositionReadout on token grids [M,N,D] -> position Y [M,2]. Returns
    fn(token_grid [*,N,D]) -> [*,2]. Label-free instrument (positions are proprioception)."""
    M, N, D = token_grids.shape
    net = SlotPositionReadout(D, num_slots, iters).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xg = token_grids.to(dev).float(); Yt = Y.to(dev).float()
    with torch.enable_grad():
        for _ in range(epochs):
            i = torch.randint(0, M, (bs,), device=dev)
            pred = net(Xg[i]); loss = ((pred - Yt[i]) ** 2).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()

    @torch.no_grad()
    def fn(zt):
        return net(zt.to(dev).float())
    return fn
