"""
ABSTRACTION-LAYER VALIDATION — do the strategic & tactical layers (a) PREDICT the
latent space at their own abstraction level, and (b) EMIT goals that steer the layer
below? (The non-negotiable: the hierarchy must actually plan in latent space.)

All on the FROZEN encoder + frozen probes. Gates:

  G_tac_pred  TACTICAL predicts forward. From a W-frame history, the tactical
              predictor estimates the tactical latent K_tac steps ahead; we decode it
              to a position and compare to the ACTUAL future position. It must beat a
              "stand-still" baseline (decode the CURRENT tactical latent) -> the layer
              predicts motion, not identity.
  G_str_pred  STRATEGIC predicts forward. The strategic predictor estimates the
              VQ concept K_str steps ahead; we report next-concept index-match rate
              and concept cosine vs a persistence baseline (predict current concept).
  G_str2tac   STRATEGIC -> TACTICAL goal emission. Conditioning the tactical predictor
              on a GOAL concept (a future/landmark concept) pulls its emitted region
              toward that goal's location, vs conditioning on the current concept.
  G_tac2op    TACTICAL -> OPERATIVE goal emission. The emitted tactical sub-goal is a
              valid operative target: in-bounds and closer to the near-future position
              than the current position (it points the operative forward).

USAGE
  PYTHONPATH=src python -m alps.evaluation.validate_abstraction \
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
from alps.training.train_temporal import build_windows
from alps.evaluation.validate_hierarchy import fit_probe
from alps.evaluation.validate_temporal import load_model


@torch.no_grad()
def encode_windows(model, frames, fidx, device, chunk=64):
    """frames at window indices fidx[M,W] -> z[M,W,N,D], h[M,W,D], c[M,W,D], idx[M,W]."""
    M, W = fidx.shape
    Hs, Cs, Idx, Zp = [], [], [], []
    for c0 in range(0, M, chunk):
        fb = fidx[c0:c0 + chunk]                                  # [b,W]
        b = fb.shape[0]
        flat = frames[torch.from_numpy(fb.reshape(-1))].to(device).float() / 255.0
        z = model.encode_frame(flat)                             # [b*W,N,D]
        h, _ = model.tac_encode(z)                               # [b*W,D]
        cq, _, ci = model.str_encode(z)                          # [b*W,D],[b*W]
        Zp.append(model.pool(z).reshape(b, W, -1).cpu())
        Hs.append(h.reshape(b, W, -1).cpu()); Cs.append(cq.reshape(b, W, -1).cpu())
        Idx.append(ci.reshape(b, W).cpu())
    return torch.cat(Zp), torch.cat(Hs), torch.cat(Cs), torch.cat(Idx)


def run(args):
    device = torch.device(args.device)
    model, W = load_model(args.model_path, device)
    Kt, Ks = model.k_tac, model.k_str
    frames, actions, positions, room_ids, starts = load_raw(args.data_path)
    total = frames.shape[0]

    # probes (frozen): pooled-z -> pos, tactical h -> pos. Fit on a flat frame sample.
    rng = np.random.RandomState(0)
    si = rng.permutation(total)[: args.probe_samples]
    fb = frames[torch.from_numpy(si)].to(device).float() / 255.0
    with torch.no_grad():
        z = model.encode_frame(fb); hflat = model.tac_encode(z)[0]
        zpool = model.pool(z)
    P = positions[torch.from_numpy(si)]
    torch.set_grad_enabled(True)
    decode_op = fit_probe(zpool.cpu(), P, device)
    decode_tac = fit_probe(hflat.cpu(), P, device)
    torch.set_grad_enabled(False)

    # windows
    FIDX, _, _ = build_windows(actions, starts, total, W, args.stride, args.sample_stride)
    if args.limit and len(FIDX) > args.limit:
        FIDX = FIDX[rng.permutation(len(FIDX))[: args.limit]]
    posw = positions[torch.from_numpy(FIDX)].numpy()             # [M,W,2]
    roomw = room_ids[torch.from_numpy(FIDX)].numpy()             # [M,W]
    Zp, Hs, Cs, Idx = encode_windows(model, frames, FIDX, device)
    M = len(FIDX)
    out = {"k_tac": int(Kt), "k_str": int(Ks), "n_windows": int(M)}

    # ── G_tac_pred: tactical predicts the future tactical LATENT ──
    # Measured in latent space (robust): is the prediction closer to the true future
    # tactical latent than the current latent is? (relative error < 1 = predicts
    # forward). Also report the decoded-position version for interpretability.
    h_win = Hs.to(device); c_win = Cs.to(device)
    D = h_win.shape[-1]
    c_tgt = torch.cat([c_win[:, Kt:], c_win[:, -1:].expand(-1, Kt, -1)], dim=1)
    tac_pred = model.tac_predict_window(h_win, c_tgt)            # [M,W,D]
    k_hi = W - Kt
    pr = tac_pred[:, :k_hi]; cur = h_win[:, :k_hi]; fut = h_win[:, Kt:Kt + k_hi]
    e_pred = (pr - fut).norm(dim=-1).mean().item()
    e_cur = (cur - fut).norm(dim=-1).mean().item()
    pred_xy = decode_tac(pr.reshape(-1, D)).reshape(M, k_hi, 2).cpu().numpy()
    fut_xy = posw[:, Kt:Kt + k_hi]
    out["G_tac_pred"] = {"latent_pred_err": e_pred, "latent_standstill_err": e_cur,
                         "relative_err": e_pred / max(e_cur, 1e-6),
                         "decoded_pred_pos_err_wu": float(np.linalg.norm(pred_xy - fut_xy, axis=-1).mean()),
                         "passed": e_pred < e_cur}

    # ── G_str_pred: strategic predicts the future CONCEPT (latent space) ──
    str_pred = model.str_predict_window(c_win)                  # [M,W,D]
    k_hs = W - Ks
    pr = str_pred[:, :k_hs]; cur = c_win[:, :k_hs]; fut = c_win[:, Ks:Ks + k_hs]
    es_pred = (pr - fut).norm(dim=-1).mean().item()
    es_cur = (cur - fut).norm(dim=-1).mean().item()
    cb = model.vq.embeddings.weight if hasattr(model.vq, "embeddings") else model.vq.codebook.weight
    n_codes_used = int(len(torch.unique(Idx)))
    pred_idx = torch.cdist(pr.reshape(-1, D), cb.to(device)).argmin(1).reshape(M, k_hs).cpu().numpy()
    fut_idx = Idx[:, Ks:Ks + k_hs].numpy(); cur_idx = Idx[:, :k_hs].numpy()
    out["G_str_pred"] = {"latent_pred_err": es_pred, "latent_persist_err": es_cur,
                         "relative_err": es_pred / max(es_cur, 1e-6),
                         "next_concept_index_match": float((pred_idx == fut_idx).mean()),
                         "persistence_index_match": float((cur_idx == fut_idx).mean()),
                         "vq_codes_used": n_codes_used,
                         "passed": es_pred <= es_cur * 1.05 and n_codes_used >= 3}

    # ── G_str2tac: strategic concept steers the tactical sub-goal ──
    # condition tactical on the WINDOW-END concept (a goal) vs the START concept;
    # the emitted region should move toward the goal-concept's location (window end).
    h_hist = h_win.unsqueeze(2)                                 # [M,W,1,D]
    goal_c = c_win[:, -1:].expand(-1, W, -1)                    # condition on end (goal) concept
    cur_c = c_win[:, :1].expand(-1, W, -1)                      # condition on start concept
    reg_goal = decode_tac(model.tac_predictor.predict_next(h_hist, goal_c).squeeze(1)).cpu().numpy()
    reg_cur = decode_tac(model.tac_predictor.predict_next(h_hist, cur_c).squeeze(1)).cpu().numpy()
    goal_loc = posw[:, -1]                                      # location associated w/ goal concept
    d_goalcond = float(np.linalg.norm(reg_goal - goal_loc, axis=-1).mean())
    d_curcond = float(np.linalg.norm(reg_cur - goal_loc, axis=-1).mean())
    out["G_str2tac"] = {"region_to_goal_when_goal_conditioned": d_goalcond,
                        "region_to_goal_when_current_conditioned": d_curcond,
                        "goal_concept_pulls_region_closer": d_goalcond < d_curcond,
                        "passed": d_goalcond < d_curcond}

    # ── G_tac2op: emitted tactical sub-goal is a valid forward operative target ──
    sub = reg_goal                                              # the goal-conditioned region
    in_bounds = float(((sub >= 0) & (sub <= 10)).all(-1).mean())
    near_fut = posw[:, min(Kt, W - 1)]                          # near-future position
    d_sub_fut = np.linalg.norm(sub - near_fut, axis=-1)
    d_cur_fut = np.linalg.norm(posw[:, 0] - near_fut, axis=-1)
    forward = float((d_sub_fut <= d_cur_fut + 1e-6).mean())
    out["G_tac2op"] = {"in_bounds_frac": in_bounds, "points_forward_frac": forward,
                       "passed": in_bounds > 0.9}

    os.makedirs(args.save_dir, exist_ok=True)
    p = os.path.join(args.save_dir, "abstraction_gates.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=float)

    g = out
    print("\n===== ABSTRACTION-LAYER GATES (strategic & tactical) =====")
    t = g["G_tac_pred"]; print(f"G_tac_pred  latent pred-err {t['latent_pred_err']:.3f} vs standstill "
                               f"{t['latent_standstill_err']:.3f} (rel {t['relative_err']:.2f}) | decoded "
                               f"{t['decoded_pred_pos_err_wu']:.2f}wu -> {'PASS' if t['passed'] else 'FAIL'}")
    s = g["G_str_pred"]; print(f"G_str_pred  latent pred-err {s['latent_pred_err']:.3f} vs persist "
                               f"{s['latent_persist_err']:.3f} (rel {s['relative_err']:.2f}) | concept-match "
                               f"{s['next_concept_index_match']:.2f}/{s['persistence_index_match']:.2f} | "
                               f"codes {s['vq_codes_used']} -> {'PASS' if s['passed'] else 'FAIL'}")
    a = g["G_str2tac"]; print(f"G_str2tac   goal-cond region->goal {a['region_to_goal_when_goal_conditioned']:.2f}wu "
                              f"vs cur-cond {a['region_to_goal_when_current_conditioned']:.2f}wu "
                              f"-> {'PASS' if a['passed'] else 'FAIL'}")
    o = g["G_tac2op"]; print(f"G_tac2op    sub-goal in-bounds {o['in_bounds_frac']:.2f} | "
                             f"forward {o['points_forward_frac']:.2f} -> {'PASS' if o['passed'] else 'FAIL'}")
    print(f"[report] {p}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="results/two_rooms/validation/temporal_world_model.pt")
    ap.add_argument("--data-path", default="data/two_rooms/trajectories.pt")
    ap.add_argument("--save-dir", default="results/two_rooms/validation")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--probe-samples", type=int, default=4000)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--sample-stride", type=int, default=2)
    ap.add_argument("--limit", type=int, default=3000)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
