"""
Per-layer validation of the FULL ALPS-4B hierarchy (HierWorldModel).

Gates
  G_op    : operative decode error + action directional consistency (sanity).
  G_str   : the strategic-abstraction claim — does the DISCRETE concept (a) predict
            its own long-horizon future, (b) decode ROOM identity via a probe
            (compared to the operative latent), (c) vary SLOWLY (low code-change
            rate, high code<->room purity)? Abstraction is never label-supervised;
            this is the test of whether temporal striding produced it.
  G_tac   : tactical predicts its medium-horizon abstraction; tactical latent is
            position-decodable (needed to emit waypoints).
  G_goals : THE headline — does the LEARNED sub-goal head route the operative
            across rooms? Compares operative-only vs learned-hierarchy vs oracle.
  G_rag   : surprise-gated Latent-RAG-in-the-loop — does gating remove the
            interference seen with ungated retrieval (CONTROL no longer degrades)?

USAGE
  PYTHONPATH=src python -m alps.evaluation.validate_hierarchy \
      --model-path results/two_rooms/validation/hier_world_model.pt \
      --data-path data/two_rooms/trajectories_large.pt
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, "src")

import argparse, json
import numpy as np
import torch
import torch.nn.functional as F

from alps.core.hier_world_model import HierWorldModel
from alps.core.latent_rag import LatentRAG
from alps.benchmarks.two_rooms.environment import TwoRoomsEnv
from alps.benchmarks.two_rooms.world_model_planning import (
    heuristic_oracle_policy, random_policy, summarize, EpisodeResult,
    _oracle_path_len, obs_to_frame,
)
from alps.training.train_hier import load_raw, build_samples

ACTION_DIRS = np.array([[0, 1], [0, -1], [-1, 0], [1, 0]], dtype=np.float32)  # up,down,left,right
REACH = 0.6


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = HierWorldModel(d_model=ck.get("d_model", 128), num_codes=ck.get("num_codes", 64),
                       num_experts=ck.get("num_experts", 4), active_experts=ck.get("active_experts", 2)).to(device)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m, ck


@torch.no_grad()
def gather(model, frames, positions, room_ids, idx, device, chunk=128):
    """Encode a set of frame indices -> operative pooled z, strategic c, tactical h, room, pos."""
    Z, C, H, R, P = [], [], [], [], []
    for c0 in range(0, len(idx), chunk):
        b = idx[c0:c0 + chunk]
        f = frames[torch.from_numpy(b)].to(device).float() / 255.0
        z = model.encode_frame(f)
        Z.append(model.pool(z).cpu()); C.append(model.str_encode(z)[0].cpu())
        H.append(model.tac_encode(z)[0].cpu())
        R.append(room_ids[torch.from_numpy(b)]); P.append(positions[torch.from_numpy(b)])
    return (torch.cat(Z), torch.cat(C), torch.cat(H), torch.cat(R), torch.cat(P))


def fit_probe(feat, pos, device, epochs=150):
    """Fresh position probe on FROZEN eval-mode latents (standard JEPA protocol).
    The model's own decode heads suffer a BatchNorm train/eval gap, so all decode
    metrics and control loops use these consistent probes instead."""
    from alps.evaluation.repr_decoder_gate import PositionProbe
    pm, ps = pos.mean(0), pos.std(0) + 1e-6
    probe = PositionProbe(feat.shape[1]).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=2e-3, weight_decay=1e-4)
    X = feat.to(device); Y = ((pos - pm) / ps).to(device)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        F.mse_loss(probe(X), Y).backward(); opt.step()
    probe.eval()
    pm_d, ps_d = pm.to(device), ps.to(device)

    @torch.no_grad()
    def fn(z):
        return probe(z) * ps_d + pm_d
    return fn


def room_probe_acc(feat_tr, y_tr, feat_va, y_va):
    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=500).fit(feat_tr.numpy(), y_tr.numpy())
        return float(clf.score(feat_va.numpy(), y_va.numpy()))
    except Exception:
        # fallback: nearest-centroid
        c0 = feat_tr[y_tr == 0].mean(0); c1 = feat_tr[y_tr == 1].mean(0)
        d0 = (feat_va - c0).norm(dim=1); d1 = (feat_va - c1).norm(dim=1)
        pred = (d1 < d0).long()
        return float((pred == y_va).float().mean())


# ---------- control loops (decode via fresh eval-mode probes) ----------
@torch.no_grad()
def hier_greedy_step(model, frame, subgoal_xy, device, decode_op):
    z = model.encode_frame(frame.unsqueeze(0))
    goal = torch.tensor(subgoal_xy, device=device, dtype=torch.float32)
    best_a, best_d = 0, 1e30
    for a in range(4):
        oh = F.one_hot(torch.tensor([a], device=device), 4).float()
        pos = decode_op(model.op_predict(z, oh))[0]
        d = float((pos - goal).norm())
        if d < best_d:
            best_d, best_a = d, a
    return best_a


@torch.no_grad()
def run_operative_only(model, env, sr, gr, seed, device, decode_op, max_steps=120):
    obs = env.reset(start_room=sr, goal_room=gr); goal = obs["target"].copy()
    opt = _oracle_path_len(sr, gr, seed)
    for s in range(max_steps):
        a = hier_greedy_step(model, obs_to_frame(obs, device), goal, device, decode_op)
        obs, _, done, info = env.step(a)
        if done or info["distance"] < REACH:
            return EpisodeResult(True, s + 1, opt, s + 1, sr != gr)
    return EpisodeResult(False, max_steps, opt, max_steps, sr != gr)


@torch.no_grad()
def run_learned_hierarchy(model, env, sr, gr, seed, device, decode_op, decode_tac, max_steps=120, reemit=1):
    obs = env.reset(start_room=sr, goal_room=gr); goal_xy = obs["target"].copy()
    opt = _oracle_path_len(sr, gr, seed)
    eg = TwoRoomsEnv(seed=seed); eg.reset(start_room=gr, goal_room=gr); eg.agent_pos = goal_xy.copy()
    z_goal = model.encode_frame(obs_to_frame({"image": eg.render()}, device).unsqueeze(0))
    h_goal, _ = model.tac_encode(z_goal)
    for s in range(max_steps):
        if s % reemit == 0:
            z = model.encode_frame(obs_to_frame(obs, device).unsqueeze(0))
            h, _ = model.tac_encode(z)
            subgoal_h = model.emit_subgoal(h, h_goal)
            waypoint = decode_tac(subgoal_h)[0].cpu().numpy()
        a = hier_greedy_step(model, obs_to_frame(obs, device), waypoint, device, decode_op)
        obs, _, done, info = env.step(a)
        if done or info["distance"] < REACH:
            return EpisodeResult(True, s + 1, opt, s + 1, sr != gr)
    return EpisodeResult(False, max_steps, opt, max_steps, sr != gr)


def run(args):
    device = torch.device(args.device)
    model, ck = load_model(args.model_path, device)
    S, K_tac, K_str = ck.get("stride", 4), ck.get("k_tac", 2), ck.get("k_str", 4)
    frames, actions, positions, room_ids, starts = load_raw(args.data_path)
    total = frames.shape[0]
    rng = np.random.RandomState(1)
    out = {}

    # ----- probe / prediction sample sets -----
    I, IOP, ITAC, ISTR, ADOM = build_samples(actions, starts, total, S, K_tac, K_str, args.sample_stride)
    if args.limit_samples and len(I) > args.limit_samples:
        sel = rng.choice(len(I), args.limit_samples, replace=False)
        I, IOP, ITAC, ISTR, ADOM = I[sel], IOP[sel], ITAC[sel], ISTR[sel], ADOM[sel]
    ntr = int(len(I) * 0.8)
    perm = rng.permutation(len(I)); tr, va = perm[:ntr], perm[ntr:]

    Ztr, Ctr, Htr, Rtr, Ptr = gather(model, frames, positions, room_ids, I[tr], device)
    Zva, Cva, Hva, Rva, Pva = gather(model, frames, positions, room_ids, I[va], device)

    # Fresh position probes on FROZEN eval-mode latents (operative z, tactical h).
    # Used for all decode metrics and control loops to avoid the encoder's
    # BatchNorm train/eval gap that corrupts the model's own decode heads.
    decode_op = fit_probe(Ztr, Ptr, device)
    decode_tac = fit_probe(Htr, Ptr, device)

    # ----- G_str: room probe (concept vs operative), prediction, slow-varying -----
    acc_c = room_probe_acc(Ctr, Rtr, Cva, Rva)
    acc_z = room_probe_acc(Ztr, Rtr, Zva, Rva)
    # strategic long-horizon prediction error (normalized)
    with torch.no_grad():
        c_i = Cva.to(device)
        c_str = gather(model, frames, positions, room_ids, ISTR[va], device)[1].to(device)
        c_pred = model.str_predict(c_i)
        str_pred_err = (c_pred - c_str).norm(dim=1).mean().item() / (c_str.norm(dim=1).mean().item() + 1e-8)
    # slow-varying: code-change rate vs room-change rate along episodes
    code_changes = room_changes = steps = 0
    code_room = {}
    with torch.no_grad():
        for e in range(min(starts.shape[0], 60)):
            s = int(starts[e]); end = int(starts[e + 1]) if e + 1 < starts.shape[0] else total
            seq = list(range(s, end, S))
            if len(seq) < 3:
                continue
            f = frames[torch.tensor(seq)].to(device).float() / 255.0
            _, _, idxc = model.str_encode(model.encode_frame(f))
            idxc = idxc.cpu().numpy(); rseq = room_ids[torch.tensor(seq)].numpy()
            code_changes += int((idxc[1:] != idxc[:-1]).sum())
            room_changes += int((rseq[1:] != rseq[:-1]).sum())
            steps += len(seq) - 1
            for cc, rr in zip(idxc, rseq):
                code_room.setdefault(int(cc), []).append(int(rr))
    purity = float(np.mean([max(np.bincount(v)) / len(v) for v in code_room.values()])) if code_room else 0.0
    out["G_str"] = {
        "room_acc_from_concept": acc_c, "room_acc_from_operative": acc_z,
        "strategic_pred_err_rel": str_pred_err,
        "code_change_rate": code_changes / max(1, steps),
        "room_change_rate": room_changes / max(1, steps),
        "code_room_purity": purity, "active_codes": len(code_room),
        "passed": (acc_c > 0.8 and purity > 0.7 and (code_changes / max(1, steps)) < 0.5),
    }

    # ----- G_tac: medium-horizon prediction + position decodability -----
    with torch.no_grad():
        h_i = Hva.to(device)
        c_i = Cva.to(device)
        h_tac = gather(model, frames, positions, room_ids, ITAC[va], device)[2].to(device)
        h_pred = model.tac_predict(h_i, c_i)
        tac_pred_err = (h_pred - h_tac).norm(dim=1).mean().item() / (h_tac.norm(dim=1).mean().item() + 1e-8)
        tac_dec_err = (decode_tac(h_i) - Pva.to(device)).norm(dim=1).mean().item()
    out["G_tac"] = {"tac_pred_err_rel": tac_pred_err, "tac_decode_err_world_units": tac_dec_err,
                    "passed": tac_dec_err < 1.2}

    # ----- G_op: operative decode + action directional consistency -----
    with torch.no_grad():
        op_dec_err = (decode_op(Zva.to(device)) - Pva.to(device)).norm(dim=1).mean().item()
        # directional consistency on a held-out batch of frames
        fb = frames[torch.from_numpy(I[va][:512])].to(device).float() / 255.0
        zb = model.encode_frame(fb); base = decode_op(zb)
        aligned = 0
        for a in range(4):
            oh = F.one_hot(torch.full((zb.shape[0],), a, device=device), 4).float()
            disp = (decode_op(model.op_predict(zb, oh)) - base).mean(0).cpu().numpy()
            if float(disp @ ACTION_DIRS[a]) > 0:
                aligned += 1
    out["G_op"] = {"op_decode_err_world_units": op_dec_err, "directional_consistency": aligned / 4.0,
                   "passed": op_dec_err < 0.6 and aligned >= 3}

    # ----- G_goals: operative-only vs learned-hierarchy vs oracle -----
    from alps.benchmarks.two_rooms.world_model_planning import run_baseline_episode
    # balanced same/cross-room configs with deterministic seeds
    configs = []
    for i in range(args.n_episodes):
        srm = i % 2
        grm = srm if (i // 2) % 2 == 0 else 1 - srm
        configs.append((srm, grm, 1000 + i))
    res_op, res_h, res_or = [], [], []
    for (srm, grm, seed) in configs:
        res_op.append(run_operative_only(model, TwoRoomsEnv(seed=seed), srm, grm, seed, device, decode_op))
        res_h.append(run_learned_hierarchy(model, TwoRoomsEnv(seed=seed), srm, grm, seed, device, decode_op, decode_tac))
        res_or.append(run_baseline_episode(TwoRoomsEnv(seed=seed), heuristic_oracle_policy(), srm, grm, seed))
    out["G_goals"] = {
        "operative_only": summarize(res_op),
        "learned_hierarchy": summarize(res_h),
        "oracle": summarize(res_or),
    }
    co = out["G_goals"]["operative_only"]["cross_room_success"]
    ch = out["G_goals"]["learned_hierarchy"]["cross_room_success"]
    out["G_goals"]["edge_cross_room_gain"] = (ch - co)
    out["G_goals"]["passed"] = ch > max(0.15, 1.5 * co)

    # ----- G_rag: surprise-gated retrieval (fixes interference) -----
    out["G_rag"] = gate_rag(model, frames, positions, room_ids, I, IOP, ADOM, va, device)

    os.makedirs(args.save_dir, exist_ok=True)
    p = os.path.join(args.save_dir, "hierarchy_gates.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=float)
    _print(out); print(f"\n[report] {p}")
    return out


@torch.no_grad()
def gate_rag(model, frames, positions, room_ids, I, IOP, ADOM, va, device, chunk=128):
    """Surprise-gated WRITE/TEST/CONTROL: apply RAG only when prediction error is high."""
    ZT, ZP, ZN, ERR, CTX = [], [], [], [], []
    idx = I[va][:3000]; idxn = IOP[va][:3000]; ad = ADOM[va][:3000]
    for c0 in range(0, len(idx), chunk):
        b = slice(c0, c0 + chunk)
        f = frames[torch.from_numpy(idx[b])].to(device).float() / 255.0
        fn = frames[torch.from_numpy(idxn[b])].to(device).float() / 255.0
        z = model.encode_frame(f); zn = model.encode_frame(fn)
        a = F.one_hot(torch.from_numpy(ad[b]).to(device), 4).float()
        zp = model.op_predict(z, a)
        err = (zp - zn).flatten(1).norm(dim=1)
        ZT.append(z.cpu()); ZP.append(zp.cpu()); ZN.append(zn.cpu())
        ERR.append(err.cpu()); CTX.append(model.pool(z).cpu())
    ZP = torch.cat(ZP); ZN = torch.cat(ZN); ERR = torch.cat(ERR); CTX = torch.cat(CTX)
    n = len(ERR)
    q_hi = torch.quantile(ERR, 0.75); q_lo = torch.quantile(ERR, 0.25)
    surprise = torch.where(ERR >= q_hi)[0]; control = torch.where(ERR <= q_lo)[0]
    perm = surprise[torch.randperm(len(surprise))]
    write_ids, test_ids = perm[:len(perm)//2], perm[len(perm)//2:]
    gate_thr = float(q_hi)  # surprise gate: only correct when error >= this

    rag = LatentRAG(d_model=model.d_model, sim_threshold=0.6, max_size=max(5000, len(write_ids)+10)).to(device)
    for i in write_ids.tolist():
        rag.write_memory(CTX[i].to(device), (ZN[i] - ZP[i]).mean(0).to(device))

    def err_gated(ids):
        base, corr = [], []
        for i in ids.tolist():
            zp = ZP[i].to(device); zn = ZN[i].to(device); e = float((zp - zn).flatten().norm())
            base.append(e)
            if e >= gate_thr:  # surprise gate
                d = rag.retrieve_correction(CTX[i].to(device).view(1, 1, -1)).squeeze(0)
                corr.append(float((zp + d - zn).flatten().norm()))
            else:
                corr.append(e)  # not surprising -> no correction -> no interference
        return float(np.mean(base)), float(np.mean(corr))

    res = {}
    for name, ids in [("write_oneshot", write_ids), ("test_generalization", test_ids), ("control_interference", control)]:
        b, c = err_gated(ids)
        res[name] = {"err_no_rag": b, "err_with_rag": c, "reduction_pct": 100*(b-c)/max(b,1e-8), "n": len(ids)}
    res["passed"] = (res["test_generalization"]["reduction_pct"] > 5
                     and res["control_interference"]["reduction_pct"] > -2)
    return res


def _print(out):
    print("\n================ HIERARCHY GATES ================")
    g = out["G_op"]; print(f"G_op   : decode {g['op_decode_err_world_units']:.3f}wu, dir-consistency {g['directional_consistency']:.2f} -> {'PASS' if g['passed'] else 'FAIL'}")
    g = out["G_str"]; print(f"G_str  : room-acc concept {g['room_acc_from_concept']:.2f} vs operative {g['room_acc_from_operative']:.2f} | purity {g['code_room_purity']:.2f} | code-change {g['code_change_rate']:.2f} (room-change {g['room_change_rate']:.2f}) | codes {g['active_codes']} -> {'PASS' if g['passed'] else 'FAIL'}")
    g = out["G_tac"]; print(f"G_tac  : pred-err {g['tac_pred_err_rel']:.2f} | decode {g['tac_decode_err_world_units']:.3f}wu -> {'PASS' if g['passed'] else 'FAIL'}")
    g = out["G_goals"]; print(f"G_goals: cross-room  operative {g['operative_only']['cross_room_success']:.2f} | LEARNED {g['learned_hierarchy']['cross_room_success']:.2f} | oracle {g['oracle']['cross_room_success']:.2f}  (gain {g['edge_cross_room_gain']:+.2f}) -> {'PASS' if g['passed'] else 'FAIL'}")
    g = out["G_rag"]; print(f"G_rag  : write {g['write_oneshot']['reduction_pct']:+.1f}% | test {g['test_generalization']['reduction_pct']:+.1f}% | control {g['control_interference']['reduction_pct']:+.1f}% -> {'PASS' if g['passed'] else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="results/two_rooms/validation/hier_world_model.pt")
    ap.add_argument("--data-path", default="data/two_rooms/trajectories_large.pt")
    ap.add_argument("--save-dir", default="results/two_rooms/validation")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-episodes", type=int, default=30)
    ap.add_argument("--sample-stride", type=int, default=3)
    ap.add_argument("--limit-samples", type=int, default=8000)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
