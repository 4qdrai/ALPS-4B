"""
ALPS-4B Representation & Decoder Validation — Acceptance Gates G1 and G2.

This module implements the Proposal A.1 root-cause fixes and their falsifiable
acceptance gates. It is the prerequisite for any planning result: if the latent
space is not position-decodable (G1) and the action-conditioned predictor does
not move the latent in a decodable, action-dependent way (G2), then latent-space
planning cannot work, regardless of compute.

ROOT-CAUSE FIXES (applied in `train` mode):
  1. Real temporal input            — `frame_skip` defaults to 4 so consecutive
                                       clip frames differ by ~1.2 world units
                                       instead of ~0.3 (a ~4 px move in 128 px).
  2. Position-aware latent           — auxiliary position loss is ON by default,
                                       directly supervising spatial decodability.
  3. Action-conditioned world model  — operative predictor (z_t, a_t) -> z_{t+1}
                                       trained on real env transitions.

GATES:
  G1 (decoder gate):
      Freeze the encoder, train an INDEPENDENT position probe on held-out data,
      report the mean Euclidean decoding error in WORLD UNITS (not latent MSE).
      PASS if  err < --g1-threshold  (default 0.3 world units).

  G2 (action-sensitivity + dynamics decodability):
      * action_sensitivity = mean pairwise ||pred(z,a_i) - pred(z,a_j)|| over the
        4 discrete actions.
      * one_step_pred_error = ||pred(z,a_true) - z_next_true||.
      * ratio = action_sensitivity / one_step_pred_error.
        PASS if ratio > --g2-ratio (default 2.0): actions must move the latent
        substantially more than the residual prediction error, otherwise CEM has
        no signal to optimize.
      * dynamics_decode_error = ||decode(pred(z,a_true)) - true_next_xy||  (world
        units). A working world model keeps this near the static decode error and
        FAR below a degenerate identity predictor (which would equal the step
        displacement).
      * directional_consistency: does each action move the DECODED position in the
        intended direction (up=+y, down=-y, left=-x, right=+x)?

USAGE
    # Measure the shipped model (the "before"):
    python -m alps.evaluation.repr_decoder_gate probe-existing \
        --ckpt results/two_rooms/two_rooms_model.pt

    # Apply the fixes and re-measure (the "after"):
    python -m alps.evaluation.repr_decoder_gate train \
        --epochs 30 --frame-skip 4

Outputs JSON + figures under results/two_rooms/validation/.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, "src")

import argparse
import json
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alps.core.encoders import VisionEncoder
from alps.core.predictor import MultiScalePredictor
from alps.core.sigreg import SIGReg
from alps.benchmarks.two_rooms.dataset import TwoRoomsDataset


# Intended (dx, dy) for each discrete action (matches TwoRoomsEnv.ACTION_DELTAS sign).
#   0=up(+y), 1=down(-y), 2=left(-x), 3=right(+x)
ACTION_DIRECTIONS = torch.tensor(
    [[0.0, 1.0], [0.0, -1.0], [-1.0, 0.0], [1.0, 0.0]], dtype=torch.float32
)
ACTION_NAMES = ["up", "down", "left", "right"]
SINGLE_FRAME_T = 8  # broadcast a single frame to a pseudo-clip for the 3D ViT


# ════════════════════════════════════════════════════════════════════════════
#  Minimal action-conditioned world model (encoder + operative predictor + decoder)
#  This is exactly "rung 1/2" of the ablation ladder — reusable for planning later.
# ════════════════════════════════════════════════════════════════════════════

class ReprWorldModel(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        enc_depth: int = 4,
        enc_heads: int = 4,
        patch_size: tuple = (2, 16, 16),
        max_patches: int = 512,
        pred_depth: int = 6,
        d_action: int = 4,
        lambda_sigreg: float = 0.1,
        sigreg_slices: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.encoder = VisionEncoder(
            d_model=d_model, depth=enc_depth, num_heads=enc_heads,
            patch_size=patch_size, max_patches=max_patches,
        )
        self.predictor = MultiScalePredictor(
            d_model=d_model, d_cond=d_action, depth=pred_depth, num_heads=enc_heads,
        )
        self.position_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2),
        )
        self.sigreg = SIGReg(d_model=d_model, num_slices=sigreg_slices)
        self.lambda_sigreg = lambda_sigreg
        # Position normalization buffers (filled from data before training).
        self.register_buffer("pos_mean", torch.tensor([5.0, 5.0]))
        self.register_buffer("pos_std", torch.tensor([3.0, 3.0]))

    def encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """frame [B,3,H,W] -> latent tokens [B,N,D] (static pseudo-clip)."""
        clip = frame.unsqueeze(2).expand(-1, -1, SINGLE_FRAME_T, -1, -1)
        return self.encoder(clip)

    def predict_next(self, z: torch.Tensor, a_onehot: torch.Tensor) -> torch.Tensor:
        return self.predictor(z, a_onehot)

    def decode_pos_norm(self, z: torch.Tensor) -> torch.Tensor:
        return self.position_head(z.mean(dim=1))  # [B,2] normalized

    def decode_pos(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode_pos_norm(z) * self.pos_std + self.pos_mean  # [B,2] world units


# ════════════════════════════════════════════════════════════════════════════
#  Independent position probe (for G1 — does NOT share weights with the model)
# ════════════════════════════════════════════════════════════════════════════

class PositionProbe(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 3:
            z = z.mean(dim=1)
        return self.net(z)


# ════════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════════

def split_dataset(dataset: TwoRoomsDataset, val_frac: float = 0.2, seed: int = 0):
    n = len(dataset)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(n * val_frac))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def latent_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-sample L2 over flattened (N,D); returns [B]."""
    return (a - b).flatten(1).norm(dim=1)


# ════════════════════════════════════════════════════════════════════════════
#  Training (applies the fixes)
# ════════════════════════════════════════════════════════════════════════════

def train_model(
    dataset: TwoRoomsDataset,
    train_idx: List[int],
    device: torch.device,
    d_model: int = 128,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    lambda_sigreg: float = 0.1,
    pos_weight: float = 1.0,
    dyn_weight: float = 1.0,
    limit_batches: Optional[int] = None,
    sigreg_slices: int = 256,
    enc_depth: int = 4,
    enc_heads: int = 4,
) -> ReprWorldModel:
    model = ReprWorldModel(d_model=d_model, enc_depth=enc_depth, enc_heads=enc_heads,
                           lambda_sigreg=lambda_sigreg, sigreg_slices=sigreg_slices).to(device)

    # Fit position normalization from the data (more stable pos-loss gradients).
    pos = dataset.positions.float()
    model.pos_mean.copy_(pos.mean(0).to(device))
    model.pos_std.copy_((pos.std(0) + 1e-6).to(device))

    sub = torch.utils.data.Subset(dataset, train_idx)
    loader = DataLoader(sub, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] ReprWorldModel params: {n_params:,} | clips(train): {len(sub)} "
          f"| batches/epoch: {len(loader)} | pos_weight={pos_weight} | sigreg={lambda_sigreg}")

    model.train()
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        agg = {"loss": 0.0, "pred": 0.0, "sig": 0.0, "pos": 0.0, "dyn": 0.0, "nb": 0}
        for bi, batch in enumerate(loader):
            if limit_batches is not None and bi >= limit_batches:
                break
            frames = batch["video_frames"].to(device)        # [B,3,T,H,W]
            acts = batch["actions_onehot"].to(device)         # [B,T,4]
            positions = batch["positions"].to(device)         # [B,T,2]
            B, _, T = frames.shape[0], frames.shape[1], frames.shape[2]

            # Encode every frame in ONE batched call (B*T) instead of a Python
            # loop of T calls — same FLOPs, far less kernel-launch overhead.
            flat = frames.permute(0, 2, 1, 3, 4).reshape(B * T, frames.shape[1],
                                                          frames.shape[3], frames.shape[4])
            z_all = model.encode_frame(flat)                      # [B*T, N, D]
            z_all = z_all.reshape(B, T, z_all.shape[1], z_all.shape[2])
            zs = [z_all[:, t] for t in range(T)]
            n_rows = B * T * zs[0].shape[1]

            # Position auxiliary loss (forces spatial decodability into the latent).
            pos_norm = (positions - model.pos_mean) / model.pos_std  # [B,T,2]
            pos_loss = torch.stack(
                [F.mse_loss(model.decode_pos_norm(zs[t]), pos_norm[:, t]) for t in range(T)]
            ).mean()

            # Action-conditioned latent prediction + action-grounded dynamics:
            #   pred_loss : predicted next latent matches the true next latent (JEPA)
            #   dyn_loss  : DECODED next position must advance per the action -> this is
            #               what forces actions to actually move the latent (gate G2).
            pred_terms, dyn_terms = [], []
            for t in range(T - 1):
                zp = model.predict_next(zs[t], acts[:, t])
                pred_terms.append(F.mse_loss(zp, zs[t + 1].detach()))
                dyn_terms.append(F.mse_loss(model.decode_pos_norm(zp), pos_norm[:, t + 1]))
            pred_loss = torch.stack(pred_terms).mean()
            dyn_loss = torch.stack(dyn_terms).mean()

            # SIGReg normalized per-row (the raw Epps-Pulley statistic scales with N,
            # which otherwise swamps the pos/pred gradients — see sigreg.py:94).
            sig_raw = model.sigreg(torch.stack(zs, dim=1).reshape(-1, T, d_model))
            sig_loss = lambda_sigreg * sig_raw / n_rows

            loss = pred_loss + sig_loss + pos_weight * pos_loss + dyn_weight * dyn_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            agg["loss"] += loss.item(); agg["pred"] += pred_loss.item()
            agg["sig"] += float(sig_loss); agg["pos"] += pos_loss.item()
            agg["dyn"] += dyn_loss.item(); agg["nb"] += 1

        nb = max(1, agg["nb"])
        print(f"  epoch {epoch:03d}/{epochs:03d} | loss {agg['loss']/nb:.4f} "
              f"| pred {agg['pred']/nb:.4f} | sigreg {agg['sig']/nb:.4f} "
              f"| pos {agg['pos']/nb:.4f} | dyn {agg['dyn']/nb:.4f} | {time.perf_counter()-t0:.1f}s")
    model.eval()
    return model


# ════════════════════════════════════════════════════════════════════════════
#  Adapter so a loaded TwoRoomsALPS checkpoint exposes the same interface
# ════════════════════════════════════════════════════════════════════════════

class ExistingModelAdapter(nn.Module):
    """Wraps a trained TwoRoomsALPS so the gates can run unmodified."""

    def __init__(self, alps_model: nn.Module, device: torch.device):
        super().__init__()
        self.m = alps_model
        self.d_model = alps_model.d_model
        self.register_buffer("pos_mean", torch.tensor([5.0, 5.0], device=device))
        self.register_buffer("pos_std", torch.tensor([3.0, 3.0], device=device))

    def encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        return self.m.encode_single_frame(frame)

    def predict_next(self, z: torch.Tensor, a_onehot: torch.Tensor) -> torch.Tensor:
        z_op, _ = self.m.operative_layer(z, torch.zeros_like(z))
        return self.m.operative_layer.predict_next_state(z_op, a_onehot)

    def decode_pos(self, z: torch.Tensor) -> torch.Tensor:
        # Uses the model's own (possibly untrained) position head, raw world units.
        return self.m.predict_position(z)


def load_existing(ckpt_path: str, device: torch.device,
                  d_model: int = 128, num_embeddings: int = 64,
                  num_experts: int = 4, active_experts: int = 2) -> ExistingModelAdapter:
    from alps.benchmarks.two_rooms.train_two_rooms import TwoRoomsALPS
    model = TwoRoomsALPS(
        d_model=d_model, d_action=4, num_embeddings=num_embeddings,
        num_experts=num_experts, active_experts=active_experts,
        encoder_depth=4, encoder_num_heads=4, encoder_patch_size=(2, 16, 16),
        encoder_max_patches=512,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = ckpt.get("model_state_dict", ckpt)
    # Some checkpoints ship a LayerNorm (not BatchNorm) projection head.
    if "encoder.projection_head.0.weight" in sd:
        model.encoder.projection_head = nn.Sequential(
            nn.Linear(model.d_model, model.d_model),
            nn.LayerNorm(model.d_model), nn.GELU(),
        ).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] {ckpt_path}: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return ExistingModelAdapter(model, device)


# ════════════════════════════════════════════════════════════════════════════
#  Gate G1 — decoder gate (independent probe, held-out, world units)
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _encode_pairs(model, dataset, idx, device, max_pairs=8000):
    """Collect (pooled latent, world-position) pairs from clip frames."""
    zs, ys = [], []
    count = 0
    for i in idx:
        sample = dataset[i]
        frames = sample["video_frames"].to(device)   # [3,T,H,W]
        positions = sample["positions"]              # [T,2]
        T = frames.shape[1]
        z = model.encode_frame(frames.permute(1, 0, 2, 3))  # [T,3,H,W] -> [T,N,D]
        zs.append(z.mean(dim=1).cpu())                       # [T,D]
        ys.append(positions)
        count += T
        if count >= max_pairs:
            break
    return torch.cat(zs, 0), torch.cat(ys, 0)


def gate_g1(model, dataset, train_idx, val_idx, device,
            probe_epochs=120, threshold=0.3) -> Dict:
    print("\n[G1] Decoder gate - training independent position probe on frozen latents ...")
    Ztr, Ytr = _encode_pairs(model, dataset, train_idx, device)
    Zva, Yva = _encode_pairs(model, dataset, val_idx, device)

    ymean, ystd = Ytr.mean(0), Ytr.std(0) + 1e-6
    Ytr_n = (Ytr - ymean) / ystd

    probe = PositionProbe(model.d_model).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=probe_epochs)
    ds = torch.utils.data.TensorDataset(Ztr.to(device), Ytr_n.to(device))
    dl = DataLoader(ds, batch_size=256, shuffle=True)

    best_err = float("inf")
    Zva_d, Yva_d = Zva.to(device), Yva.to(device)
    for ep in range(probe_epochs):
        probe.train()
        for zb, yb in dl:
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(probe(zb), yb)
            loss.backward(); opt.step()
        sched.step()
        probe.eval()
        with torch.no_grad():
            pred = probe(Zva_d) * ystd.to(device) + ymean.to(device)
            err = (pred - Yva_d).norm(dim=1).mean().item()
        best_err = min(best_err, err)

    passed = best_err < threshold
    print(f"[G1] held-out mean decode error = {best_err:.4f} world units "
          f"(threshold {threshold}) -> {'PASS' if passed else 'FAIL'}")
    # data for the scatter figure
    with torch.no_grad():
        pred_va = (probe(Zva_d) * ystd.to(device) + ymean.to(device)).cpu().numpy()

    # Head-independent decoder usable by G2 (denormalizes the probe output).
    ymean_d, ystd_d = ymean.to(device), ystd.to(device)

    @torch.no_grad()
    def decode_fn(z: torch.Tensor) -> torch.Tensor:
        return probe(z) * ystd_d + ymean_d

    g1 = {
        "held_out_decode_error_world_units": best_err,
        "threshold": threshold,
        "passed": passed,
        "n_train_pairs": int(Ztr.shape[0]),
        "n_val_pairs": int(Zva.shape[0]),
        "_scatter_true": Yva.numpy().tolist(),
        "_scatter_pred": pred_va.tolist(),
    }
    return g1, decode_fn


# ════════════════════════════════════════════════════════════════════════════
#  Gate G2 — action sensitivity + dynamics decodability
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def gate_g2(model, dataset, val_idx, device, decode_fn, max_samples=3000, ratio_threshold=2.0) -> Dict:
    """`decode_fn(z)->[B,2] world units` is the independent G1 probe (head-independent)."""
    print("\n[G2] Action-sensitivity + dynamics decodability ...")
    # Gather consecutive (frame_t, frame_{t+1}, action_t, pos_t, pos_{t+1}) transitions.
    ft, ftn, at, pt, ptn = [], [], [], [], []
    for i in val_idx:
        s = dataset[i]
        fr = s["video_frames"]            # [3,T,H,W]
        T = fr.shape[1]
        for t in range(T - 1):
            ft.append(fr[:, t]); ftn.append(fr[:, t + 1])
            at.append(int(s["actions"][t].item()))
            pt.append(s["positions"][t]); ptn.append(s["positions"][t + 1])
        if len(ft) >= max_samples:
            break
    ft = torch.stack(ft); ftn = torch.stack(ftn)         # keep on CPU; chunk to GPU
    at = torch.tensor(at); pt = torch.stack(pt); ptn = torch.stack(ptn)
    B = ft.shape[0]

    dirs = ACTION_DIRECTIONS.to(device)
    pair_idx = [(a, b) for a in range(4) for b in range(a + 1, 4)]
    pair_sum = [0.0] * len(pair_idx)
    onestep_sum = static_sum = dyn_sum = stepdisp_sum = 0.0
    disp_sum = [torch.zeros(2, device=device) for _ in range(4)]
    N = 0

    # Encode/predict in mini-batches: a full-batch encode_frame broadcasts every
    # frame to a T=8 pseudo-clip, so [B,3,8,128,128] OOMs for large B.
    chunk = 128
    for c0 in range(0, B, chunk):
        c1 = min(c0 + chunk, B)
        bc = c1 - c0
        fchunk = ft[c0:c1].to(device); fnchunk = ftn[c0:c1].to(device)
        atc = at[c0:c1].to(device)
        ptc = pt[c0:c1].to(device); ptnc = ptn[c0:c1].to(device)

        z_t = model.encode_frame(fchunk)        # [bc,N,D]
        z_next = model.encode_frame(fnchunk)
        preds = [model.predict_next(
            z_t, F.one_hot(torch.full((bc,), a, device=device), num_classes=4).float())
            for a in range(4)]

        for pi, (a, b) in enumerate(pair_idx):
            pair_sum[pi] += latent_l2(preds[a], preds[b]).sum().item()

        z_pred_true = model.predict_next(z_t, F.one_hot(atc, num_classes=4).float())
        onestep_sum += latent_l2(z_pred_true, z_next).sum().item()

        base_xy = decode_fn(z_t)
        static_sum += (base_xy - ptc).norm(dim=1).sum().item()
        dyn_sum += (decode_fn(z_pred_true) - ptnc).norm(dim=1).sum().item()
        stepdisp_sum += (ptnc - ptc).norm(dim=1).sum().item()
        for a in range(4):
            disp_sum[a] += (decode_fn(preds[a]) - base_xy).sum(0)
        N += bc

    action_sensitivity = float(np.mean([s / N for s in pair_sum]))
    one_step_err = onestep_sum / N
    ratio = action_sensitivity / (one_step_err + 1e-8)
    static_decode_err = static_sum / N
    dyn_decode_err = dyn_sum / N
    step_disp = stepdisp_sum / N

    per_action = {}
    aligned = 0
    quiver = []
    for a in range(4):
        disp = disp_sum[a] / N                      # [2]
        intended = dirs[a]
        dn = disp / (disp.norm() + 1e-8)
        cos = float((dn * intended).sum().item())
        per_action[ACTION_NAMES[a]] = {
            "mean_decoded_dx": float(disp[0]), "mean_decoded_dy": float(disp[1]),
            "cosine_to_intended": cos,
        }
        quiver.append([float(disp[0]), float(disp[1])])
        if cos > 0:
            aligned += 1
    directional_consistency = aligned / 4.0

    passed = (ratio > ratio_threshold) and (dyn_decode_err < 0.5 * step_disp + static_decode_err) \
        and (directional_consistency >= 0.75)
    print(f"[G2] action_sensitivity={action_sensitivity:.4f} | one_step_err={one_step_err:.4f} "
          f"| ratio={ratio:.2f} (>{ratio_threshold})")
    print(f"[G2] static_decode_err={static_decode_err:.4f} | dyn_decode_err={dyn_decode_err:.4f} "
          f"| step_disp={step_disp:.4f} world units")
    print(f"[G2] directional_consistency={directional_consistency:.2f} "
          f"-> {'PASS' if passed else 'FAIL'}")
    return {
        "n_transitions": int(B),
        "action_sensitivity": action_sensitivity,
        "one_step_pred_error": one_step_err,
        "sensitivity_to_error_ratio": ratio,
        "ratio_threshold": ratio_threshold,
        "static_decode_error_world_units": static_decode_err,
        "dynamics_decode_error_world_units": dyn_decode_err,
        "expected_step_displacement_world_units": step_disp,
        "directional_consistency": directional_consistency,
        "per_action": per_action,
        "passed": passed,
        "_quiver": quiver,
    }


# ════════════════════════════════════════════════════════════════════════════
#  Figures
# ════════════════════════════════════════════════════════════════════════════

def make_figures(g1: Dict, g2: Dict, save_dir: str, tag: str):
    os.makedirs(save_dir, exist_ok=True)

    # G1 scatter: true vs decoded (x and y).
    true = np.array(g1["_scatter_true"]); pred = np.array(g1["_scatter_pred"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for k, (ax, name) in enumerate(zip(axes, ["X", "Y"])):
        ax.scatter(true[:, k], pred[:, k], s=6, alpha=0.3, color="#1e90ff")
        lim = [0, 10]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.set_xlabel(f"true {name}"); ax.set_ylabel(f"decoded {name}")
        ax.set_title(f"{name}: held-out decoding")
    fig.suptitle(f"G1 decoder gate [{tag}] — mean err "
                 f"{g1['held_out_decode_error_world_units']:.3f} world units "
                 f"({'PASS' if g1['passed'] else 'FAIL'})")
    fig.tight_layout()
    p1 = os.path.join(save_dir, f"g1_decoding_{tag}.png")
    fig.savefig(p1, dpi=150); plt.close(fig)

    # G2 quiver: per-action decoded displacement vs intended direction.
    q = np.array(g2["_quiver"])
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]
    intended = ACTION_DIRECTIONS.numpy()
    scale = max(1e-6, np.abs(q).max())
    for a in range(4):
        ax.arrow(0, 0, q[a, 0], q[a, 1], head_width=0.04 * scale, color=colors[a],
                 length_includes_head=True, lw=2, label=f"{ACTION_NAMES[a]} (decoded)")
        ax.arrow(0, 0, intended[a, 0] * 0.5 * scale, intended[a, 1] * 0.5 * scale,
                 head_width=0.03 * scale, color=colors[a], alpha=0.35,
                 length_includes_head=True, lw=1, ls=":")
    ax.axhline(0, color="#aaa", lw=0.5); ax.axvline(0, color="#aaa", lw=0.5)
    ax.set_aspect("equal"); ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"G2 per-action decoded displacement [{tag}]\n"
                 f"dir-consistency {g2['directional_consistency']:.2f} | "
                 f"sens/err ratio {g2['sensitivity_to_error_ratio']:.2f} "
                 f"({'PASS' if g2['passed'] else 'FAIL'})")
    fig.tight_layout()
    p2 = os.path.join(save_dir, f"g2_action_quiver_{tag}.png")
    fig.savefig(p2, dpi=150); plt.close(fig)
    print(f"[figures] saved {p1} and {p2}")
    return p1, p2


# ════════════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════════════

def run(args):
    device = torch.device(args.device)
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    frame_skip = args.frame_skip
    print(f"[data] loading {args.data_path} (frame_skip={frame_skip}) ...")
    dataset = TwoRoomsDataset(args.data_path, clip_length=8, stride=4, frame_skip=frame_skip)
    if args.limit_clips:
        # Deterministic subset for quick smoke runs.
        dataset.clip_indices = dataset.clip_indices[: args.limit_clips]
        print(f"[data] limited to {len(dataset)} clips (smoke).")
    train_idx, val_idx = split_dataset(dataset, val_frac=0.2, seed=0)

    if args.mode == "train":
        tag = f"trained_fs{frame_skip}"
        model = train_model(
            dataset, train_idx, device, d_model=args.d_model, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr, lambda_sigreg=args.lambda_sigreg,
            pos_weight=args.pos_weight, dyn_weight=args.dyn_weight,
            sigreg_slices=args.sigreg_slices, enc_depth=args.enc_depth, enc_heads=args.enc_heads,
            limit_batches=args.limit_batches if args.limit_batches > 0 else None,
        )
        if args.save_model:
            mp = os.path.join(save_dir, f"repr_world_model_fs{frame_skip}.pt")
            torch.save({"model_state_dict": model.state_dict(),
                        "d_model": args.d_model, "enc_depth": args.enc_depth,
                        "enc_heads": args.enc_heads, "frame_skip": frame_skip}, mp)
            print(f"[save] model -> {mp}")
    else:  # probe-existing
        tag = "existing_" + Path(args.ckpt).stem
        model = load_existing(args.ckpt, device, d_model=args.d_model,
                              num_embeddings=args.num_embeddings,
                              num_experts=args.num_experts,
                              active_experts=args.active_experts)

    g1, decode_fn = gate_g1(model, dataset, train_idx, val_idx, device,
                            probe_epochs=args.probe_epochs, threshold=args.g1_threshold)
    g2 = gate_g2(model, dataset, val_idx, device, decode_fn, ratio_threshold=args.g2_ratio)

    p1, p2 = make_figures(g1, g2, save_dir, tag)

    # Strip large arrays before writing JSON.
    g1_out = {k: v for k, v in g1.items() if not k.startswith("_")}
    g2_out = {k: v for k, v in g2.items() if not k.startswith("_")}
    report = {
        "tag": tag, "mode": args.mode, "frame_skip": frame_skip,
        "device": str(device), "d_model": args.d_model,
        "G1_decoder_gate": g1_out, "G2_world_model_gate": g2_out,
        "overall_pass": bool(g1_out["passed"] and g2_out["passed"]),
        "figures": [p1, p2],
    }
    out = os.path.join(save_dir, f"repr_decoder_gate_{tag}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 72)
    print(f"  G1 decoder gate:     {'PASS' if g1_out['passed'] else 'FAIL'} "
          f"({g1_out['held_out_decode_error_world_units']:.3f} world units)")
    print(f"  G2 world-model gate: {'PASS' if g2_out['passed'] else 'FAIL'} "
          f"(ratio {g2_out['sensitivity_to_error_ratio']:.2f}, "
          f"dyn-decode {g2_out['dynamics_decode_error_world_units']:.3f} wu)")
    print(f"  OVERALL:             {'PASS' if report['overall_pass'] else 'FAIL'}")
    print(f"  report -> {out}")
    print("=" * 72)
    return report


def build_parser():
    p = argparse.ArgumentParser(description="ALPS-4B representation & decoder acceptance gates (G1, G2)")
    sub = p.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-path", default="data/two_rooms/trajectories.pt")
    common.add_argument("--save-dir", default="results/two_rooms/validation")
    common.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    common.add_argument("--frame-skip", type=int, default=4)
    common.add_argument("--d-model", type=int, default=128)
    common.add_argument("--enc-depth", type=int, default=4, help="encoder ViT depth (12 ≈ ViT-Tiny/Small)")
    common.add_argument("--enc-heads", type=int, default=4, help="encoder attention heads (must divide d_model)")
    common.add_argument("--probe-epochs", type=int, default=120)
    common.add_argument("--g1-threshold", type=float, default=0.3)
    common.add_argument("--g2-ratio", type=float, default=2.0)
    common.add_argument("--limit-clips", type=int, default=0)

    pt = sub.add_parser("train", parents=[common])
    pt.add_argument("--epochs", type=int, default=30)
    pt.add_argument("--batch-size", type=int, default=16)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--lambda-sigreg", type=float, default=0.1)
    pt.add_argument("--sigreg-slices", type=int, default=256)
    pt.add_argument("--pos-weight", type=float, default=1.0)
    pt.add_argument("--dyn-weight", type=float, default=1.0)
    pt.add_argument("--limit-batches", type=int, default=0)
    pt.add_argument("--save-model", action="store_true")

    pe = sub.add_parser("probe-existing", parents=[common])
    pe.add_argument("--ckpt", required=True)
    pe.add_argument("--num-embeddings", type=int, default=64)
    pe.add_argument("--num-experts", type=int, default=4)
    pe.add_argument("--active-experts", type=int, default=2)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
