"""
G1 root-cause diagnostic — WHY does the unsupervised latent fail to decode position?

Runs three linear probes (closed-form ridge, fit on train / measured on held-out)
and one motion measurement, to separate the competing hypotheses for a failed G1:

  POOLED latent  (model.pool = mean over tokens)  -> reproduces the G1 number.
  FULL token grid (all N*D token features)        -> upper bound of a LINEAR read-out.
  RAW pixels      (downsampled frame)             -> control: is position even IN the input?

Interpretation:
  * pixels decode well, full-grid decodes well, pooled poorly
        -> POOLING bottleneck. The agent (1-2 of 64 tokens) is washed out by the mean.
           Fix: attention/[CLS] pooling everywhere (encode/plan/control), LeWM-style.
  * pixels decode well, full-grid ALSO poor (~pooled ~random)
        -> the ENCODER discards position. SSL prediction had no incentive to encode the
           moving agent. Fix: inter-frame motion (STRIDE/frame-skip) or a position anchor.
  * pixels ALSO poor
        -> something upstream (data/render) is wrong; position is not in the frames.

Plus: mean per-step agent displacement in WORLD UNITS at several strides, to see whether
next-frame prediction is even being asked to track the agent.

USAGE
  PYTHONPATH=src python -m alps.evaluation.diagnose_g1 \
      --model-path results/two_rooms/validation/unsupervised/unsup_temporal.pt \
      --data-path  data/two_rooms/trajectories_unsup.pt
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, "src")
import argparse
import numpy as np
import torch

from alps.evaluation.validate_temporal import load_model
from alps.training.train_hier import load_raw


@torch.no_grad()
def ridge_probe(Xtr, Ytr, Xte, Yte, lams=(0.1, 1.0, 10.0, 100.0, 1000.0)):
    """Closed-form ridge linear probe. Standardize X, fit on train, report Euclidean
    decode error (WORLD UNITS) on train and held-out, for the best lambda by TEST err.
    Returns (best_train_err, best_test_err, best_lam, r2_test)."""
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    n, d = Xtr.shape
    Xtr1 = torch.cat([Xtr, torch.ones(n, 1, device=Xtr.device)], 1)        # bias
    Xte1 = torch.cat([Xte, torch.ones(Xte.shape[0], 1, device=Xte.device)], 1)
    A0 = Xtr1.t() @ Xtr1
    b = Xtr1.t() @ Ytr
    eye = torch.eye(A0.shape[0], device=A0.device)
    eye[-1, -1] = 0.0                                                       # don't penalize bias
    best = (1e9, 1e9, None, -1e9)
    for lam in lams:
        W = torch.linalg.solve(A0 + lam * eye, b)
        ptr, pte = Xtr1 @ W, Xte1 @ W
        etr = (ptr - Ytr).norm(dim=1).mean().item()
        ete = (pte - Yte).norm(dim=1).mean().item()
        ss_res = ((pte - Yte) ** 2).sum().item()
        ss_tot = ((Yte - Yte.mean(0)) ** 2).sum().item()
        r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
        if ete < best[1]:
            best = (etr, ete, lam, r2)
    return best


@torch.no_grad()
def encode_all(model, frames, idx, device, chunk=128):
    """Return pooled [n,D] and full-grid [n,N*D] latents for the sampled frames."""
    pooled, grid = [], []
    for c0 in range(0, len(idx), chunk):
        b = idx[c0:c0 + chunk]
        fr = frames[b].to(device).float() / 255.0
        z = model.encode_frame(fr)                 # [b,N,D]
        pooled.append(z.mean(dim=1).cpu())
        grid.append(z.reshape(z.shape[0], -1).cpu())
    return torch.cat(pooled), torch.cat(grid)


def motion_stats(positions, starts, total, strides=(1, 4, 8, 12)):
    """Mean agent displacement (world units) at each stride, within episodes."""
    pos = positions.numpy()
    st = starts.numpy().tolist() + [total]
    out = {}
    for S in strides:
        d = []
        for e in range(len(st) - 1):
            a, b = st[e], st[e + 1]
            if b - a > S:
                seg = pos[a:b]
                d.append(np.linalg.norm(seg[S:] - seg[:-S], axis=1))
        out[S] = float(np.concatenate(d).mean()) if d else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="results/two_rooms/validation/unsupervised/unsup_temporal.pt")
    ap.add_argument("--data-path", default="data/two_rooms/trajectories_unsup.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-samples", type=int, default=8000)
    ap.add_argument("--pixel-size", type=int, default=16, help="downsample side for the raw-pixel control probe")
    a = ap.parse_args()
    device = torch.device(a.device)

    model, W = load_model(a.model_path, device)
    frames, actions, positions, room_ids, starts = load_raw(a.data_path)
    total = frames.shape[0]
    print(f"[diag] model={a.model_path}  d_model={model.d_model}  total_frames={total:,}")

    rng = np.random.RandomState(0)
    idx = rng.permutation(total)[: min(a.n_samples, total)]
    n_tr = int(0.8 * len(idx))
    tr, te = idx[:n_tr], idx[n_tr:]
    Y = positions.float()
    Ytr, Yte = Y[tr].to(device), Y[te].to(device)

    # --- feature set 1+2: pooled + full-grid latent ---
    pooled_tr, grid_tr = encode_all(model, frames, tr, device)
    pooled_te, grid_te = encode_all(model, frames, te, device)

    # --- feature set 3: raw downsampled pixels (control) ---
    P = a.pixel_size
    def pixels(ix):
        fr = frames[ix].float() / 255.0                       # [n,3,128,128]
        fr = torch.nn.functional.adaptive_avg_pool2d(fr, (P, P))
        return fr.reshape(fr.shape[0], -1)
    pix_tr, pix_te = pixels(tr).to(device), pixels(te).to(device)

    print(f"[diag] features: pooled D={pooled_tr.shape[1]} | grid D={grid_tr.shape[1]} | "
          f"pixels D={pix_tr.shape[1]} | n_train={len(tr)} n_test={len(te)}")

    print("\n=== LINEAR POSITION PROBE (ridge, world units, held-out) ===")
    print(f"{'features':<22}{'train_err':>10}{'test_err':>10}{'best_lam':>10}{'R2_test':>9}")
    for name, Xtr, Xte in [
        ("POOLED latent (G1)", pooled_tr.to(device), pooled_te.to(device)),
        ("FULL token grid",    grid_tr.to(device),   grid_te.to(device)),
        ("RAW pixels (control)", pix_tr,             pix_te),
    ]:
        etr, ete, lam, r2 = ridge_probe(Xtr, Ytr, Xte, Yte)
        print(f"{name:<22}{etr:>10.3f}{ete:>10.3f}{lam:>10.1f}{r2:>9.3f}")

    print("\n=== INTER-FRAME MOTION (mean |dpos| world units, within episodes) ===")
    ms = motion_stats(positions, starts, total)
    for S, v in ms.items():
        tag = "  <- training stride S=4" if S == 4 else ""
        print(f"  stride {S:>2}: {v:.3f} wu{tag}")
    print("\n[guide] world is 10x10; ~3.3 wu = random. If FULL-grid << POOLED -> pooling")
    print("        bottleneck (use attention/[CLS] pool). If FULL-grid ~ POOLED ~ random but")
    print("        PIXELS decode -> encoder discards the agent (raise STRIDE / add anchor).")


if __name__ == "__main__":
    main()
