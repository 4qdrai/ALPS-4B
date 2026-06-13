"""
Train the TemporalHierWorldModel — full ALPS hierarchy with LeWM-style K-frame
history at every scale, on windows of W consecutive frames (teacher-forced).

For a window of W frames (base stride S) the causal predictors are supervised at
every valid position:
  operative : pred[:,k] -> z[:,k+1]            (1-step dynamics, + position decode)
  tactical  : pred[:,k] -> h[:,k+K_tac]        (mid-horizon abstraction)
  strategic : pred[:,k] -> c[:,k+K_str]        (slow-horizon discrete concept)
  sub-goal  : emit(h_k, h_goal=h_{W-1}) -> h_{k+K_tac}   (hindsight goal routing)
Encoder is trained only by operative+position (stop-grad isolation); abstraction
layers learn on detached features. Collapse prevented by SIGReg (operative) +
VICReg (tactical/strategic).

USAGE
  PYTHONPATH=src python -m alps.training.train_temporal \
      --data-path data/two_rooms/trajectories_large.pt --window 6 --epochs 25 --save-model
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, "src")

import argparse, time
import numpy as np
import torch
import torch.nn.functional as F

from alps.core.temporal_world_model import TemporalHierWorldModel
from alps.training.train_hier import load_raw, var_cov_reg


def build_windows(actions, starts, total, W, S, sample_stride):
    """Return [M,W] frame-index, [M,W] dominant-action, and [M] episode-end-index
    (the last frame of the episode, for sampling variable far-horizon goals)."""
    span = (W - 1) * S
    FIDX, AIDX, GEND = [], [], []
    E = starts.shape[0]
    for e in range(E):
        s = int(starts[e]); end = int(starts[e + 1]) if e + 1 < E else total
        i = s
        while i + span < end:
            fidx = [i + k * S for k in range(W)]
            acts = []
            for k in range(W):
                blk = actions[fidx[k]: fidx[k] + S] if k < W - 1 else actions[fidx[k]:fidx[k] + 1]
                acts.append(int(torch.bincount(blk, minlength=4).argmax().item()) if len(blk) else 0)
            FIDX.append(fidx); AIDX.append(acts); GEND.append(end - 1)
            i += sample_stride * S
    return np.array(FIDX), np.array(AIDX), np.array(GEND)


def train(args):
    device = torch.device(args.device)
    if not hasattr(args, "lewm_ssl"):
        args.lewm_ssl = False
    if args.lewm_ssl:
        # LeWM-faithful research mode: pure SSL (no labels), SIGReg-only, no stop-grad,
        # no VICReg. (Open: collapses on the trivial Two-Rooms task; see SIGREG_FINDINGS.)
        args.self_supervised = True
        print("[mode] LeWM-SSL (faithful): SIGReg-only, no stop-grad, no VICReg, no labels.")
    if getattr(args, "self_supervised", False):
        args.pos_weight = 0.0
        args.dyn_weight = 0.0
        print("[mode] SELF-SUPERVISED: no position/dynamics labels on the encoder; "
              "latent read out by a frozen probe at eval.")
    else:
        print("[mode] ANCHORED HIERARCHY (default): position anchor + stop-grad target "
              "+ VICReg + SIGReg backstop -> healthy encoder for the strategic/tactical layers.")
    frames, actions, positions, room_ids, starts = load_raw(args.data_path)
    total = frames.shape[0]
    FIDX, AIDX, GEND = build_windows(actions, starts, total, args.window, args.stride, args.sample_stride)
    n = len(FIDX)
    if args.limit_samples and n > args.limit_samples:
        sel = np.random.RandomState(0).choice(n, args.limit_samples, replace=False)
        FIDX, AIDX, GEND = FIDX[sel], AIDX[sel], GEND[sel]; n = len(FIDX)
    print(f"[data] {total} frames | {n} windows (W={args.window}, S={args.stride}, "
          f"K_tac={args.k_tac}, K_str={args.k_str})")

    model = TemporalHierWorldModel(
        d_model=args.d_model, enc_depth=args.enc_depth, enc_heads=args.enc_heads,
        num_codes=args.num_codes, num_experts=args.num_experts, active_experts=args.active_experts,
        op_depth=args.op_depth, abs_depth=args.abs_depth, k_tac=args.k_tac, k_str=args.k_str,
        lambda_sigreg=args.lambda_sigreg, sigreg_slices=args.sigreg_slices,
        max_frames=args.window + 1, use_projection_head=True).to(device)
    pm, ps = positions.mean(0), positions.std(0) + 1e-6
    model.pos_mean.copy_(pm.to(device)); model.pos_std.copy_(ps.to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"[model] params {sum(p.numel() for p in model.parameters()):,} "
          f"| encoder {sum(p.numel() for p in model.encoder.parameters()):,}")

    W, Kt, Ks = args.window, args.k_tac, args.k_str
    rng = np.random.RandomState(0)
    model.train()
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        order = rng.permutation(n)
        agg = {k: 0.0 for k in ["loss", "op", "dyn", "pos", "tac", "str", "vq", "sub", "sig", "col"]}
        nb = 0
        for b0 in range(0, n - args.batch_size + 1, args.batch_size):
            bw = FIDX[order[b0:b0 + args.batch_size]]      # [B,W]
            ba = AIDX[order[b0:b0 + args.batch_size]]      # [B,W]
            bg = GEND[order[b0:b0 + args.batch_size]]      # [B] episode-end frame idx
            B = bw.shape[0]
            fl = torch.from_numpy(bw.reshape(-1))
            fr = frames[fl].to(device).float() / 255.0     # [B*W,3,H,W]
            z = model.encode_frame(fr)                     # [B*W,N,D]
            N, D = z.shape[1], z.shape[2]
            z = z.reshape(B, W, N, D)
            pos = positions[fl].to(device).reshape(B, W, 2)
            pos_n = (pos - model.pos_mean) / model.pos_std
            a = F.one_hot(torch.from_numpy(ba).to(device), 4).float()   # [B,W,4]

            # operative (trains encoder); pos decoded from per-frame pooled tokens
            op_pred = model.op_predict_window(z, a)                      # [B,W,N,D]
            # `--lewm-ssl` = LeWM-faithful (2603.19312, Eq.3): NO stop-gradient on the
            # target (SIGReg alone prevents collapse). DEFAULT (anchored hierarchy):
            # stop-grad the target + position anchor + VICReg = the validated healthy
            # config the strategic/tactical layers train on. (`--stopgrad-target`
            # forces stop-grad even under --lewm-ssl, for diagnostics.)
            _stopgrad = (not args.lewm_ssl) or getattr(args, "stopgrad_target", False)
            _tgt = z[:, 1:W].detach() if _stopgrad else z[:, 1:W]
            L_op = F.mse_loss(op_pred[:, :W-1], _tgt)
            L_pos = F.mse_loss(model.pos_head(z.mean(dim=2)), pos_n)             # decode z[:,k]->pos_k
            L_dyn = F.mse_loss(model.pos_head(op_pred[:, :W-1].mean(dim=2)), pos_n[:, 1:W])

            # stop-grad isolation for abstraction layers
            zd = z.detach()
            c_list, vq_tot, h_list, moe_tot = [], 0.0, [], 0.0
            for k in range(W):
                c_k, vq_k, _ = model.str_encode(zd[:, k]); c_list.append(c_k); vq_tot = vq_tot + vq_k
                h_k, moe_k = model.tac_encode(zd[:, k]); h_list.append(h_k); moe_tot = moe_tot + moe_k
            c_win = torch.stack(c_list, dim=1)              # [B,W,D]
            h_win = torch.stack(h_list, dim=1)              # [B,W,D]

            str_pred = model.str_predict_window(c_win)
            L_str = F.mse_loss(str_pred[:, :W-Ks], c_win[:, Ks:].detach()) if W > Ks else torch.zeros((), device=device)
            # GOAL-CONDITIONED tactical: condition on the TARGET strategic concept
            # (the concept K_tac steps ahead = where we're heading), so at inference
            # the tactical emits a rough sub-goal region toward the NEXT strategic
            # landmark. Robust because strategic concepts are coarse/discrete (unlike
            # the far continuous goal that broke the sub-goal head).
            c_tgt = torch.cat([c_win[:, Kt:], c_win[:, -1:].expand(-1, Kt, -1)], dim=1)  # cond[k]=c[k+Kt]
            tac_pred = model.tac_predict_window(h_win, c_tgt.detach())
            L_tac = F.mse_loss(tac_pred[:, :W-Kt], h_win[:, Kt:].detach()) if W > Kt else torch.zeros((), device=device)
            L_tacpos = F.mse_loss(model.tac_pos_head(h_win), pos_n)

            # hindsight goal-conditioned sub-goal with VARIABLE FAR-HORIZON goals:
            # sample a goal frame beyond the window on the same episode, so the head
            # is trained for distant goals (the inference regime) — fixes the OOD
            # failure where it was only ever trained on window-end goals.
            last = bw[:, -1]
            off = rng.randint(1, args.goal_max + 1, size=B) * args.stride
            goal_idx = np.minimum(last + off, bg)
            gframe = frames[torch.from_numpy(goal_idx)].to(device).float() / 255.0
            with torch.no_grad():
                h_goal = model.tac_encode(model.encode_frame(gframe))[0]      # [B,D]
            subgoal = model.emit_subgoal(h_win, h_goal.unsqueeze(1).expand(-1, W, -1))   # [B,W,D]
            L_sub = F.mse_loss(subgoal[:, :W-Kt], h_win[:, Kt:].detach()) if W > Kt else torch.zeros((), device=device)

            # ── anti-collapse ───────────────────────────────────────────────────
            z_pool = z.mean(dim=2).reshape(B * W, D)
            h_flat = h_win.reshape(B * W, D)
            s_flat = model.str_pre(zd.reshape(B * W, N, D))
            if args.lewm_ssl:
                # LeWM Eq.3 across the FULL hierarchy (not just the operative): SIGReg on
                # each scale's EMBEDDINGS *and* its PREDICTIONS (z_t, z_{t+1}, AND z-hat),
                # the predictions passed through that scale's BatchNorm predictor-projector
                # (LeWM "the predictor is also followed by a projector"). SIGReg-only, no
                # EMA, no stop-grad. This keeps operative + tactical + strategic all
                # collapse-free and predicting in latent space.
                op_pred_pool = model.op_pred_proj(op_pred.mean(dim=2).reshape(B * W, D))
                tac_pred_p = model.tac_pred_proj(tac_pred.reshape(B * W, D))
                str_pred_p = model.str_pred_proj(str_pred.reshape(B * W, D))
                L_sig = model.lambda_sigreg * (
                    model.sigreg(z_pool) + model.sigreg(h_flat) + model.sigreg(s_flat)
                    + model.sigreg(op_pred_pool) + model.sigreg(tac_pred_p) + model.sigreg(str_pred_p))
                L_col = torch.zeros((), device=device)
            else:
                # Anchored hierarchy (validated): per-row-normalized SIGReg as a light
                # backstop + VICReg variance/covariance floor; position anchor does the
                # heavy lifting. This is the healthy config for the abstraction layers.
                nr = B * W
                L_sig = model.lambda_sigreg * (model.sigreg(z_pool) + model.sigreg(h_flat) + model.sigreg(s_flat)) / nr
                L_col = var_cov_reg(z_pool) + var_cov_reg(h_flat) + var_cov_reg(s_flat)

            loss = (L_op + args.dyn_weight * L_dyn + args.pos_weight * (L_pos + L_tacpos)
                    + L_tac + L_str + vq_tot + L_sub + L_sig + 0.01 * moe_tot
                    + args.collapse_weight * L_col)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

            agg["loss"] += loss.item(); agg["op"] += L_op.item(); agg["dyn"] += L_dyn.item()
            agg["pos"] += L_pos.item(); agg["tac"] += float(L_tac); agg["str"] += float(L_str)
            agg["vq"] += float(vq_tot); agg["sub"] += float(L_sub)
            agg["sig"] += float(L_sig); nb += 1

        nb = max(1, nb)
        print(f"  ep {epoch:03d}/{args.epochs:03d} | loss {agg['loss']/nb:.3f} | op {agg['op']/nb:.3f} "
              f"dyn {agg['dyn']/nb:.3f} pos {agg['pos']/nb:.3f} tac {agg['tac']/nb:.3f} "
              f"str {agg['str']/nb:.3f} vq {agg['vq']/nb:.3f} sub {agg['sub']/nb:.3f} "
              f"sig {agg['sig']/nb:.3f} | {time.perf_counter()-t0:.1f}s")

    if args.save_model:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "d_model": args.d_model,
                    "enc_depth": args.enc_depth, "enc_heads": args.enc_heads,
                    "num_codes": args.num_codes, "num_experts": args.num_experts,
                    "active_experts": args.active_experts, "op_depth": args.op_depth,
                    "abs_depth": args.abs_depth, "window": args.window, "stride": args.stride,
                    "k_tac": args.k_tac, "k_str": args.k_str,
                    "use_projection_head": True}, args.out)
        print(f"[save] {args.out}")
    return model


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="data/two_rooms/trajectories_large.pt")
    ap.add_argument("--out", default="results/two_rooms/validation/temporal_world_model.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--enc-depth", type=int, default=10)
    ap.add_argument("--enc-heads", type=int, default=8)
    ap.add_argument("--op-depth", type=int, default=6)
    ap.add_argument("--abs-depth", type=int, default=4)
    ap.add_argument("--num-codes", type=int, default=64)
    ap.add_argument("--num-experts", type=int, default=4)
    ap.add_argument("--active-experts", type=int, default=2)
    ap.add_argument("--window", type=int, default=6)         # K-frame history window
    ap.add_argument("--stride", type=int, default=4)         # base stride S (env frames/op-step)
    ap.add_argument("--k-tac", type=int, default=2)
    ap.add_argument("--k-str", type=int, default=4)
    ap.add_argument("--goal-max", type=int, default=12, help="max far-goal offset (op-steps) for sub-goal training")
    ap.add_argument("--sample-stride", type=int, default=2)
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--dyn-weight", type=float, default=1.0)
    ap.add_argument("--lewm-ssl", action="store_true",
                    help="LeWM-faithful research mode: pure SSL, SIGReg-only, NO stop-grad, "
                         "NO VICReg (one mechanism). Default OFF = anchored hierarchy "
                         "(healthy encoder for the strategic/tactical layers).")
    ap.add_argument("--stopgrad-target", action="store_true",
                    help="diagnostic: force stop-gradient on the operative target even "
                         "under --lewm-ssl")
    ap.add_argument("--self-supervised", action="store_true",
                    help="LeWM-faithful: zero position/dynamics supervision; encoder learns "
                         "only from feature prediction + collapse prevention (probe at eval)")
    ap.add_argument("--collapse-weight", type=float, default=1.0)
    ap.add_argument("--lambda-sigreg", type=float, default=0.1)
    ap.add_argument("--sigreg-slices", type=int, default=256)
    ap.add_argument("--limit-samples", type=int, default=0)
    ap.add_argument("--save-model", action="store_true")
    return ap


def main():
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
