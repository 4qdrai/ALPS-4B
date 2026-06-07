"""
Train the FULL ALPS-4B hierarchy (HierWorldModel) with multi-scale strided
prediction, goal-conditioned sub-goal emission, and the foundation fixes.

Temporal structure (base stride S so one "operative step" shows real motion):
  operative : frame_i  -(dominant action over [i, i+S))->  frame_{i+S}
  tactical  : predict the tactical abstraction K_tac operative-steps ahead
  strategic : predict the discrete concept   K_str operative-steps ahead
  goal head : hindsight relabeling — the far-future state (i + K_str*S) is the
              GOAL; the sub-goal head learns (h_i, h_goal) -> h_{i+K_tac*S}, i.e.
              "next tactical sub-goal toward the goal". Cross-room sub-trajectories
              teach door-routing for free.

Abstraction is created purely by temporal striding (predict far ahead), NOT by
label supervision; whether the strategic concept encodes room is tested later
by a probe (validate_hierarchy.py).

USAGE
  PYTHONPATH=src python -m alps.training.train_hier \
      --data-path data/two_rooms/trajectories_large.pt --epochs 40 --save-model
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, "src")

import argparse, json, time
import numpy as np
import torch
import torch.nn.functional as F

from alps.core.hier_world_model import HierWorldModel


def var_cov_reg(x):
    """VICReg-style anti-collapse on a pooled batch [B, D]: keep per-dim variance
    near 1 and decorrelate dimensions. Prevents the tactical/strategic latents
    from collapsing (which would make strided prediction trivially solvable)."""
    x = x - x.mean(0)
    std = torch.sqrt(x.var(0) + 1e-4)
    var_loss = torch.mean(F.relu(1.0 - std))
    B, D = x.shape
    cov = (x.t() @ x) / max(1, B - 1)
    cov_off = cov - torch.diag(torch.diag(cov))
    cov_loss = (cov_off ** 2).sum() / D
    return var_loss + 0.04 * cov_loss


def load_raw(path):
    d = torch.load(path, map_location="cpu", weights_only=True)
    frames = d.get("frames", d.get("observations"))
    starts = d["episode_starts"]
    starts = torch.tensor(starts) if isinstance(starts, list) else starts
    return frames, d["actions"].long(), d["positions"].float(), d["room_ids"].long(), starts.long()


def build_samples(actions, starts, total, S, K_tac, K_str, sample_stride):
    """Return arrays of indices (i, i_op, i_tac, i_str) and dominant action a_i."""
    span = K_str * S
    I, IOP, ITAC, ISTR, ADOM = [], [], [], [], []
    E = starts.shape[0]
    for e in range(E):
        s = int(starts[e].item())
        end = int(starts[e + 1].item()) if e + 1 < E else total
        i = s
        while i + span < end:
            block = actions[i:i + S]
            a_dom = int(torch.bincount(block, minlength=4).argmax().item()) if len(block) else int(actions[i])
            I.append(i); IOP.append(i + S); ITAC.append(i + K_tac * S); ISTR.append(i + K_str * S); ADOM.append(a_dom)
            i += sample_stride * S
    return (np.array(I), np.array(IOP), np.array(ITAC), np.array(ISTR), np.array(ADOM))


def train(args):
    device = torch.device(args.device)
    frames, actions, positions, room_ids, starts = load_raw(args.data_path)
    total = frames.shape[0]
    print(f"[data] {total} frames, {starts.shape[0]} episodes")

    I, IOP, ITAC, ISTR, ADOM = build_samples(actions, starts, total, args.stride,
                                             args.k_tac, args.k_str, args.sample_stride)
    n = len(I)
    if args.limit_samples and n > args.limit_samples:
        sel = np.random.RandomState(0).choice(n, args.limit_samples, replace=False)
        I, IOP, ITAC, ISTR, ADOM = I[sel], IOP[sel], ITAC[sel], ISTR[sel], ADOM[sel]
        n = len(I)
    print(f"[data] {n} multi-scale samples (S={args.stride}, K_tac={args.k_tac}, K_str={args.k_str})")

    model = HierWorldModel(d_model=args.d_model, num_codes=args.num_codes,
                           num_experts=args.num_experts, active_experts=args.active_experts,
                           enc_depth=args.enc_depth, enc_heads=args.enc_heads,
                           lambda_sigreg=args.lambda_sigreg, sigreg_slices=args.sigreg_slices).to(device)
    pm = positions.mean(0); ps = positions.std(0) + 1e-6
    model.pos_mean.copy_(pm.to(device)); model.pos_std.copy_(ps.to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"[model] params {sum(p.numel() for p in model.parameters()):,}")

    def enc(idx_np):
        f = frames[torch.from_numpy(idx_np)].to(device).float() / 255.0   # [B,3,128,128]
        return model.encode_frame(f)

    def posn(idx_np):
        p = positions[torch.from_numpy(idx_np)].to(device)
        return (p - model.pos_mean) / model.pos_std

    rng = np.random.RandomState(0)
    model.train()
    log = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        order = rng.permutation(n)
        agg = {k: 0.0 for k in ["loss", "op", "dyn", "pos", "tac", "str", "vq", "sub", "sig", "col"]}
        nb = 0
        for b0 in range(0, n - args.batch_size + 1, args.batch_size):
            bi = order[b0:b0 + args.batch_size]
            z_i = enc(I[bi]); z_op = enc(IOP[bi]); z_tac = enc(ITAC[bi]); z_str = enc(ISTR[bi])
            a = F.one_hot(torch.from_numpy(ADOM[bi]).to(device), 4).float()

            # operative — the ONLY objectives that train the shared encoder
            # (operative latent prediction + position grounding). This protects
            # the control-critical representation from the higher-layer losses.
            op_pred = model.op_predict(z_i, a)
            L_op = F.mse_loss(op_pred, z_op.detach())
            L_dyn = F.mse_loss(model.decode_pos_norm(op_pred), posn(IOP[bi]))
            L_pos = F.mse_loss(model.decode_pos_norm(z_i), posn(I[bi]))

            # ── STOP-GRADIENT ISOLATION (ALPS design principle) ──
            # Tactical & strategic layers learn on top of DETACHED encoder features,
            # so their objectives never corrupt the operative representation.
            z_i_d, z_tac_d, z_str_d = z_i.detach(), z_tac.detach(), z_str.detach()
            # strategic
            c_i, vq_i, _ = model.str_encode(z_i_d)
            c_str, _, _ = model.str_encode(z_str_d)
            c_pred = model.str_predict(c_i)
            L_str = F.mse_loss(c_pred, c_str.detach())
            # tactical
            h_i, moe_i = model.tac_encode(z_i_d)
            h_tac, _ = model.tac_encode(z_tac_d)
            h_pred = model.tac_predict(h_i, c_i.detach())
            L_tac = F.mse_loss(h_pred, h_tac.detach())
            L_tacpos = F.mse_loss(model.tac_decode_pos_norm(h_i), posn(I[bi]))
            # goal-conditioned sub-goal (hindsight: goal = far-future state z_str)
            h_goal, _ = model.tac_encode(z_str_d)
            subgoal = model.emit_subgoal(h_i, h_goal.detach())
            L_sub = F.mse_loss(subgoal, h_tac.detach())
            # collapse prevention: SIGReg on operative tokens (normalized per row)
            # + VICReg-style variance/covariance on the pooled tactical & strategic
            # latents (otherwise strided prediction collapses to a constant).
            n_rows = z_i.shape[0] * z_i.shape[1]
            L_sig = model.lambda_sigreg * model.sigreg(z_i) / n_rows
            L_collapse = var_cov_reg(h_i) + var_cov_reg(model.str_pre(z_i_d))

            loss = (L_op + L_dyn + args.pos_weight * (L_pos + L_tacpos)
                    + L_tac + L_str + vq_i + L_sub + L_sig + 0.01 * moe_i
                    + args.collapse_weight * L_collapse)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            agg["loss"] += loss.item(); agg["op"] += L_op.item(); agg["dyn"] += L_dyn.item()
            agg["pos"] += L_pos.item(); agg["tac"] += L_tac.item(); agg["str"] += L_str.item()
            agg["vq"] += float(vq_i); agg["sub"] += L_sub.item(); agg["sig"] += float(L_sig)
            agg["col"] += float(L_collapse); nb += 1

        nb = max(1, nb)
        row = {k: v / nb for k, v in agg.items()}; row["epoch"] = epoch
        log.append(row)
        print(f"  ep {epoch:03d}/{args.epochs:03d} | loss {row['loss']:.3f} | op {row['op']:.3f} "
              f"dyn {row['dyn']:.3f} pos {row['pos']:.3f} tac {row['tac']:.3f} str {row['str']:.3f} "
              f"vq {row['vq']:.3f} sub {row['sub']:.3f} col {row['col']:.3f} | {time.perf_counter()-t0:.1f}s")

    if args.save_model:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "d_model": args.d_model,
                    "num_codes": args.num_codes, "num_experts": args.num_experts,
                    "active_experts": args.active_experts, "enc_depth": args.enc_depth,
                    "enc_heads": args.enc_heads, "stride": args.stride,
                    "k_tac": args.k_tac, "k_str": args.k_str, "log": log}, args.out)
        print(f"[save] {args.out}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="data/two_rooms/trajectories_large.pt")
    ap.add_argument("--out", default="results/two_rooms/validation/hier_world_model.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--num-codes", type=int, default=64)
    ap.add_argument("--num-experts", type=int, default=4)
    ap.add_argument("--active-experts", type=int, default=2)
    ap.add_argument("--enc-depth", type=int, default=4, help="encoder ViT depth (12 ≈ ViT-Tiny/Small)")
    ap.add_argument("--enc-heads", type=int, default=4, help="encoder heads (must divide d_model)")
    ap.add_argument("--stride", type=int, default=4)          # base stride S (env frames per op-step)
    ap.add_argument("--k-tac", type=int, default=2)           # tactical horizon in op-steps
    ap.add_argument("--k-str", type=int, default=4)           # strategic/goal horizon in op-steps
    ap.add_argument("--sample-stride", type=int, default=2)   # subsample start indices (in op-steps)
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--collapse-weight", type=float, default=1.0)
    ap.add_argument("--lambda-sigreg", type=float, default=0.1)
    ap.add_argument("--sigreg-slices", type=int, default=256)
    ap.add_argument("--limit-samples", type=int, default=0)
    ap.add_argument("--save-model", action="store_true")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
