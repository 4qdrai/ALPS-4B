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
        # PER-SLOT learned inits (not one shared mu): breaks slot symmetry without needing the
        # sampled noise, so eval can be DETERMINISTIC (an instrument must not be stochastic).
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, dim) * 0.1)
        self.slots_logsigma = nn.Parameter(torch.zeros(1, num_slots, dim))
        self.to_q = nn.Linear(dim, dim); self.to_k = nn.Linear(dim, dim); self.to_v = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.norm_in = nn.LayerNorm(dim); self.norm_slots = nn.LayerNorm(dim); self.norm_mlp = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, slots_init: torch.Tensor = None) -> torch.Tensor:
        """x [B,N,D] -> slots [B,K,D]. `slots_init` (SAVi-style recurrent binding): initialize
        from the PREVIOUS frame's slots so slot IDENTITY is consistent across a sequence — the
        v1 per-frame (stochastic) init let identities permute frame-to-frame, which turns the
        slot-dynamics prediction target into a moving target (measured: rising op loss)."""
        B, N, D = x.shape
        x = self.norm_in(x); k = self.to_k(x); v = self.to_v(x)
        if slots_init is not None:
            slots = slots_init
        else:
            mu = self.slots_mu.expand(B, -1, -1)
            if self.training:
                slots = mu + self.slots_logsigma.exp().expand(B, -1, -1) * torch.randn_like(mu)
            else:
                slots = mu                                    # deterministic at eval; symmetry is
                # broken by the per-slot learned inits, not by sampling
        for _ in range(self.iters):
            q = self.to_q(self.norm_slots(slots))
            attn = torch.softmax((q @ k.transpose(1, 2)) * self.scale, dim=1)     # softmax over SLOTS (compete)
            attn = attn / (attn.sum(-1, keepdim=True) + 1e-8)                     # normalise over tokens
            updates = attn @ v                                                    # [B,K,D]
            slots = self.gru(updates.reshape(-1, D), slots.reshape(-1, D)).reshape(B, self.num_slots, D)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots


class SlotFeatureDecoder(nn.Module):
    """Spatial-broadcast decoder: slots -> reconstructed token grid (+ per-token alpha masks).
    THE objective that ORGANIZES slot attention into object binding (Slot Attention is always
    trained with reconstruction; position regression alone gives NO binding signal -- measured:
    the probe stalls at predict-the-mean, G1 ~2.1, at 400 AND 3000 steps). DINOSAUR-style:
    reconstruct the ENCODER'S OWN FEATURES, not pixels -> stays decoder-free w.r.t. pixels and
    fully self-supervised (the target is the model's own token grid)."""
    def __init__(self, dim: int, n_tokens: int, hidden: int = None):
        super().__init__()
        hidden = hidden or 2 * dim
        self.pos = nn.Parameter(torch.randn(1, 1, n_tokens, dim) * 0.02)          # per-token query
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim + 1))

    def forward(self, slots: torch.Tensor):                   # [B,K,D] -> recon [B,N,D], masks [B,K,N]
        x = slots.unsqueeze(2) + self.pos                     # [B,K,N,D] broadcast slot to tokens
        out = self.mlp(x)                                     # [B,K,N,D+1]
        feats, alpha = out[..., :-1], out[..., -1]
        w = torch.softmax(alpha, dim=1)                       # tokens are explained by competing slots
        return (w.unsqueeze(-1) * feats).sum(1), w


class SlotPositionReadout(nn.Module):
    """Slot attention -> per-slot position + agent-score -> soft-selected AGENT position [.., 2]."""
    def __init__(self, dim: int, num_slots: int = 4, iters: int = 3):
        super().__init__()
        self.slots = SlotAttention(dim, num_slots, iters)
        self.pos_head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 2))
        self.agent_head = nn.Linear(dim, 1)

    def slots_of(self, x: torch.Tensor) -> torch.Tensor:      # [B,N,D] -> [B,K,D]
        return self.slots(x)

    def pos_of(self, s: torch.Tensor) -> torch.Tensor:        # [B,K,D] -> [B,2]
        pos = self.pos_head(s)                                                    # [B,K,2]
        w = torch.softmax(self.agent_head(s).squeeze(-1), dim=-1)                 # [B,K] soft agent selector
        return (w.unsqueeze(-1) * pos).sum(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # [.., N, D] -> [.., 2]
        shp = x.shape[:-2]
        s = self.slots_of(x.reshape(-1, x.shape[-2], x.shape[-1]))                # [B,K,D]
        return self.pos_of(s).reshape(*shp, 2)


@torch.no_grad()
def _grids(m, frames, idx, dev, bs=128):
    from alps.evaluation.validate_temporal import _gather_token_grids
    return _gather_token_grids(m, frames, idx, dev)


class EqSlotProbe(nn.Module):
    """PERMUTATION-EQUIVARIANT position probe for MODEL slots: a shared per-slot position head
    + a shared agent-selector, soft-summed -> slot ORDER cannot matter. The flat (concatenated)
    ridge is permutation-SENSITIVE: slot assignment is arbitrary per episode, so the same probe
    weights read different slots across episodes -> it mismeasured v2 by 2.3x (G1 1.21 flat vs
    0.533 equivariant on identical slot states)."""
    def __init__(self, dim: int):
        super().__init__()
        self.pos = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 2))
        self.sel = nn.Linear(dim, 1)

    def forward(self, S: torch.Tensor) -> torch.Tensor:      # [..,K,D] -> [..,2]
        shp = S.shape[:-2]
        S = S.reshape(-1, S.shape[-2], S.shape[-1])
        w = torch.softmax(self.sel(S).squeeze(-1), dim=-1)   # soft agent-slot selector
        return (w.unsqueeze(-1) * self.pos(S)).sum(1).reshape(*shp, 2)


def fit_eq_slot_probe(S: torch.Tensor, Y: torch.Tensor, dev, steps=1500, bs=256, lr=3e-3):
    """Fit a frozen EqSlotProbe on slot states S [M,K,D] -> positions Y [M,2]. Label-free
    instrument (positions are proprioception). Returns fn([..,K,D]) -> [..,2]."""
    net = EqSlotProbe(S.shape[-1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    S = S.to(dev).float(); Y = Y.to(dev).float()
    with torch.enable_grad():
        for _ in range(steps):
            i = torch.randint(0, len(S), (bs,), device=dev)
            loss = ((net(S[i]) - Y[i]) ** 2).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()

    @torch.no_grad()
    def fn(s):
        return net(s.to(dev).float())
    return fn


def fit_slot_decode(token_grids: torch.Tensor, Y: torch.Tensor, dev, num_slots=4, iters=3,
                    epochs=300, lr=3e-3, bs=256, recon_weight=1.0):
    """Fit a frozen SlotPositionReadout on token grids [M,N,D] -> position Y [M,2]. Returns
    fn(token_grid [*,N,D]) -> [*,2]. Label-free instrument (positions are proprioception).

    Trained with TWO objectives: (1) FEATURE RECONSTRUCTION of the token grid from the slots
    (spatial-broadcast decoder) -- the standard signal WITHOUT WHICH slot attention never
    organizes into objects (position-only fits stall at predict-the-mean, measured G1 ~2.1 at
    400 and 3000 steps alike); (2) the position regression through the soft agent-selector.
    Both self-supervised (targets = the model's own features + proprioceptive positions)."""
    M, N, D = token_grids.shape
    net = SlotPositionReadout(D, num_slots, iters).to(dev)
    decoder = SlotFeatureDecoder(D, N).to(dev)
    opt = torch.optim.Adam(list(net.parameters()) + list(decoder.parameters()), lr=lr)
    Xg = token_grids.to(dev).float(); Yt = Y.to(dev).float()
    net.train()
    with torch.enable_grad():
        for _ in range(epochs):
            i = torch.randint(0, M, (bs,), device=dev)
            xi = Xg[i]
            s = net.slots_of(xi)                                        # [b,K,D]
            recon, _ = decoder(s)
            l_rec = ((recon - xi) ** 2).mean()
            l_pos = ((net.pos_of(s) - Yt[i]) ** 2).sum(1).mean()
            loss = l_pos + recon_weight * l_rec
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()

    @torch.no_grad()
    def fn(zt):
        return net(zt.to(dev).float())
    return fn
