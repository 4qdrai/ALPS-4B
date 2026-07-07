"""
MoE EXPERT SPECIALIZATION (gate H11) — do the tactical experts solve different
problems, and does knocking one out hurt ITS problem selectively?

Protocol (docs/EDGE_PROGRAM.md WS-J), all on the frozen model + offline dataset
(closed-loop knockout at scale is the A40 extension):

  1. ROUTING: log the tactical router's top-1 expert per frame (eval-mode,
     deterministic — router noise is train-only).
  2. REGIMES: label each frame by problem regime from env state — MEASUREMENT
     ONLY, never used by the model:
       simple : door_zone / room0_transit / room1_transit
       complex: door_zone / transit_nokey / transit_key
     (door centers: simple (5,5); complex (5,5) locked + (2.5,5), (7.5,5) open)
  3. SPECIALIZATION: mutual information MI(top-1 expert ; regime) against a
     permutation null (label shuffles) -> z-score / p-value, plus the
     P(expert | regime) usage matrix.
  4. CAUSAL KNOCKOUT: for each expert e, mask it from routing and measure the
     per-regime increase in tactical position-decode error (frozen probe fitted
     on the FULL model's tactical latents). Specialization = the degradation
     matrix is diagonal-dominant: removing e hurts most in the regime e serves.

USAGE
  PYTHONPATH=src python -m alps.evaluation.moe_specialization \
      --model-path results/two_rooms/validation/temporal_world_model.pt \
      --data-path data/two_rooms/trajectories.pt
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, "src")
import argparse, json
import numpy as np
import torch

from alps.training.train_hier import load_raw
from alps.evaluation.validate_hierarchy import fit_probe
from alps.evaluation.validate_temporal import load_model, load_has_keys

DOORS_SIMPLE = np.array([[5.0, 5.0]], dtype=np.float32)
DOORS_COMPLEX = np.array([[5.0, 5.0], [2.5, 5.0], [7.5, 5.0]], dtype=np.float32)


def regime_labels(positions, room_ids, has_keys, complex_mode, door_r=1.2):
    """Per-frame problem-regime label (ints) + names. Measurement only."""
    P = positions.numpy() if torch.is_tensor(positions) else positions
    doors = DOORS_COMPLEX if complex_mode else DOORS_SIMPLE
    near_door = (np.linalg.norm(P[:, None, :] - doors[None, :, :], axis=2) < door_r).any(1)
    if complex_mode:
        hk = (has_keys.numpy() if torch.is_tensor(has_keys) else has_keys) > 0.5
        names = ["door_zone", "transit_nokey", "transit_key"]
        lab = np.where(near_door, 0, np.where(hk, 2, 1))
    else:
        rid = room_ids.numpy() if torch.is_tensor(room_ids) else room_ids
        names = ["door_zone", "room0_transit", "room1_transit"]
        lab = np.where(near_door, 0, np.where(rid == 0, 1, 2))
    return lab.astype(np.int64), names


def mutual_information(a, b, na, nb):
    """Plug-in MI (nats) of two integer label arrays."""
    joint = np.zeros((na, nb))
    np.add.at(joint, (a, b), 1.0)
    joint /= joint.sum()
    pa, pb = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = joint * np.log(joint / (pa @ pb))
    return float(np.nansum(t))


@torch.no_grad()
def collect(model, frames, idx, device, chunk=128, mask=None):
    """Pooled z, tactical h, and top-1 expert per frame (optionally with a
    knockout mask applied to the router)."""
    model.moe.expert_mask = mask
    model.moe.record_routing = mask is None       # record routing only on the full model
    Z, H, E = [], [], []
    for c0 in range(0, len(idx), chunk):
        b = idx[c0:c0 + chunk]
        f = frames[torch.from_numpy(b)].to(device).float() / 255.0
        z = model.encode_frame(f)
        h, _ = model.tac_encode(z)
        Z.append(model.pool(z).cpu()); H.append(h.cpu())
        if mask is None:
            E.append(model.moe.last_routing["indices"][:, 0].clone())
    model.moe.expert_mask = None
    model.moe.record_routing = False
    return torch.cat(Z), torch.cat(H), (torch.cat(E).numpy() if E else None)


def run(args):
    device = torch.device(args.device)
    model, W = load_model(args.model_path, device)
    frames, actions, positions, room_ids, starts = load_raw(args.data_path)
    hk = load_has_keys(args.data_path) if args.complex else None
    total = frames.shape[0]
    rng = np.random.RandomState(1)
    idx = rng.permutation(total)[: args.limit_samples] if args.limit_samples else rng.permutation(total)

    lab_all, names = regime_labels(positions, room_ids, hk, args.complex)
    lab = lab_all[idx]
    nE, nR = model.moe.num_experts, len(names)

    # 1-2. routing + regimes on the FULL model
    Z, Hfull, top1 = collect(model, frames, idx, device)
    P = positions[torch.from_numpy(idx)]

    # 3. MI vs permutation null
    mi = mutual_information(top1, lab, nE, nR)
    null = np.array([mutual_information(np.random.RandomState(k).permutation(top1), lab, nE, nR)
                     for k in range(args.n_perm)])
    z = float((mi - null.mean()) / (null.std() + 1e-12))
    pval = float((null >= mi).mean())
    usage = np.zeros((nE, nR))
    np.add.at(usage, (top1, lab), 1.0)
    usage_pr = usage / np.clip(usage.sum(0, keepdims=True), 1, None)   # P(expert | regime)
    expert_home = usage_pr.argmax(1)                                   # each expert's main regime

    # 4. causal knockout: per-regime tactical decode degradation (frozen probe on FULL h)
    torch.set_grad_enabled(True)
    decode_tac = fit_probe(Hfull, P, device)
    torch.set_grad_enabled(False)
    def regime_err(H):
        e = (decode_tac(H.to(device)) - P.to(device)).norm(dim=1).cpu().numpy()
        return np.array([float(e[lab == r].mean()) if (lab == r).any() else np.nan for r in range(nR)])
    base = regime_err(Hfull)
    delta = np.zeros((nE, nR))
    for e_idx in range(nE):
        mask = torch.ones(nE, dtype=torch.bool); mask[e_idx] = False
        _, Hk, _ = collect(model, frames, idx, device, mask=mask)
        delta[e_idx] = regime_err(Hk) - base

    # diagonal dominance: each expert's worst-hit regime == its most-served regime
    hit_regime = delta.argmax(1)
    used = [e for e in range(nE) if usage[e].sum() > 0.01 * len(idx)]   # experts actually used
    diag_frac = float(np.mean([hit_regime[e] == expert_home[e] for e in used])) if used else 0.0

    out = {
        "regimes": names, "n_frames": int(len(idx)),
        "expert_usage_overall": (usage.sum(1) / usage.sum()).tolist(),
        "usage_P_expert_given_regime": usage_pr.tolist(),
        "MI_nats": mi, "MI_null_mean": float(null.mean()), "MI_z": z, "MI_p": pval,
        "base_err_per_regime_wu": base.tolist(),
        "knockout_delta_err_wu": delta.tolist(),
        "expert_home_regime": [names[int(h)] for h in expert_home],
        "knockout_hit_regime": [names[int(h)] for h in hit_regime],
        "experts_used": used, "diagonal_dominance_frac": diag_frac,
        "H11_passed": bool(pval < 0.05 and z > 3 and diag_frac >= 0.5 and len(used) >= 2),
    }
    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "moe_specialization_complex.json" if args.complex
                        else "moe_specialization.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)

    print("\n===== MoE EXPERT SPECIALIZATION (H11) =====")
    print(f"experts used: {used} of {nE} | overall usage {np.round(usage.sum(1)/usage.sum(), 2).tolist()}")
    print(f"MI(expert; regime) {mi:.4f} nats | null {null.mean():.4f} | z {z:.1f} | p {pval:.3f}")
    for e in range(nE):
        print(f"  expert {e}: home={names[int(expert_home[e])]:<14s} "
              f"P(e|regime)={np.round(usage_pr[e], 2).tolist()} "
              f"knockout-hit={names[int(hit_regime[e])]:<14s} dErr={np.round(delta[e], 3).tolist()}")
    print(f"diagonal dominance {diag_frac:.2f} -> H11 {'PASS' if out['H11_passed'] else 'FAIL'}")
    print(f"[report] {path}")

    if getattr(args, "figure", False):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(4 + 1.1 * nR, 1.5 + 0.5 * nE))
        # LEFT: routing specialization P(expert | regime); RIGHT: causal knockout delta-error.
        for k, (M, ttl, cmap) in enumerate([
                (usage_pr, "routing  P(expert | regime)", "viridis"),
                (delta, "causal  knockout dErr (wu)", "magma")]):
            im = ax[k].imshow(M, aspect="auto", cmap=cmap)
            ax[k].set_xticks(range(nR)); ax[k].set_xticklabels(names, rotation=35, ha="right", fontsize=8)
            ax[k].set_yticks(range(nE)); ax[k].set_yticklabels([f"E{e}" for e in range(nE)], fontsize=8)
            ax[k].set_title(ttl, fontsize=9); fig.colorbar(im, ax=ax[k], fraction=0.046)
            for e in range(nE):                      # mark each expert's home regime
                ax[k].add_patch(plt.Rectangle((expert_home[e] - .5, e - .5), 1, 1,
                                              fill=False, edgecolor="cyan", lw=1.6))
        fig.suptitle(f"MoE expert specialization (H11)  |  MI {mi:.3f} nats  z {z:.1f}  "
                     f"p {pval:.3f}  |  diagonal dominance {diag_frac:.2f}  "
                     f"-> {'PASS' if out['H11_passed'] else 'FAIL'}", fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fpath = os.path.join(args.save_dir, "moe_specialization_complex.png" if args.complex
                             else "moe_specialization.png")
        fig.savefig(fpath, dpi=130); plt.close(fig)
        print(f"[figure] {fpath}  (cyan box = each expert's home regime; diagonal = specialization)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="results/two_rooms/validation/temporal_world_model.pt")
    ap.add_argument("--data-path", default="data/two_rooms/trajectories.pt")
    ap.add_argument("--save-dir", default="results/two_rooms/validation")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit-samples", type=int, default=5000)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--complex", action="store_true")
    ap.add_argument("--figure", action="store_true",
                    help="render the expert-specialization figure (routing heatmap + causal "
                         "knockout heatmap) as a PNG alongside the JSON (H11 made visible).")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
