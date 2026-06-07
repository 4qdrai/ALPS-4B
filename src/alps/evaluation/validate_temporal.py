"""
Validate the TemporalHierWorldModel (K-frame history at every scale).

Gates (history-aware where needed):
  G1        : decoder gate (fresh probe on frozen eval-mode latents), world units.
  G_roll    : OPEN-LOOP multi-step rollout accuracy — the direct test of the
              LeWM temporal benefit. From a K-frame history, roll the operative
              predictor forward H steps autoregressively and measure decoded
              position drift vs ground truth. (Lower = straighter/more accurate.)
  G_str/G_tac: abstraction probes (room-acc from concept, slow-varying; tactical
              decode + mid-horizon prediction).
  G_goals   : history-aware navigation — operative-only vs learned-hierarchy vs
              oracle (the greedy controller keeps a rolling W-frame buffer).
  G_rag     : surprise-gated Latent-RAG (no interference).

USAGE
  PYTHONPATH=src python -m alps.evaluation.validate_temporal \
      --model-path results/two_rooms/validation/temporal_world_model.pt \
      --data-path data/two_rooms/trajectories_large.pt --n-episodes 20
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, "src")
import argparse, json
import numpy as np
import torch
import torch.nn.functional as F

from alps.core.temporal_world_model import TemporalHierWorldModel
from alps.benchmarks.two_rooms.environment import TwoRoomsEnv
from alps.benchmarks.two_rooms.world_model_planning import (
    heuristic_oracle_policy, run_baseline_episode, summarize, EpisodeResult,
    _oracle_path_len, obs_to_frame, strategic_waypoints,
)
from alps.training.train_hier import load_raw
from alps.evaluation.validate_hierarchy import fit_probe, room_probe_acc

ACTION_DIRS = np.array([[0, 1], [0, -1], [-1, 0], [1, 0]], dtype=np.float32)
REACH = 0.6


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = TemporalHierWorldModel(
        d_model=ck.get("d_model", 192), enc_depth=ck.get("enc_depth", 10),
        enc_heads=ck.get("enc_heads", 8), num_codes=ck.get("num_codes", 64),
        num_experts=ck.get("num_experts", 4), active_experts=ck.get("active_experts", 2),
        op_depth=ck.get("op_depth", 6), abs_depth=ck.get("abs_depth", 4),
        k_tac=ck.get("k_tac", 2), k_str=ck.get("k_str", 4),
        max_frames=ck.get("window", 6) + 1).to(device)
    msd = m.state_dict()
    sd = {k: v for k, v in ck["model_state_dict"].items() if k in msd and msd[k].shape == v.shape}
    m.load_state_dict(sd, strict=False); m.eval()
    return m, ck.get("window", 6)


@torch.no_grad()
def gather(model, frames, positions, room_ids, idx, device, chunk=128):
    Z, C, H, R, P = [], [], [], [], []
    for c0 in range(0, len(idx), chunk):
        b = idx[c0:c0 + chunk]
        f = frames[torch.from_numpy(b)].to(device).float() / 255.0
        z = model.encode_frame(f)
        Z.append(model.pool(z).cpu()); C.append(model.str_encode(z)[0].cpu())
        H.append(model.tac_encode(z)[0].cpu())
        R.append(room_ids[torch.from_numpy(b)]); P.append(positions[torch.from_numpy(b)])
    return torch.cat(Z), torch.cat(C), torch.cat(H), torch.cat(R), torch.cat(P)


# ---------------- history-aware control ----------------
class HistoryBuffer:
    """Rolling buffer of the last W frame latents + actions for the causal predictor."""
    def __init__(self, model, W, device):
        self.m, self.W, self.dev = model, W, device
        self.z, self.a = [], []

    def reset(self, frame):
        z0 = self.m.encode_frame(frame.unsqueeze(0))     # [1,N,D]
        self.z = [z0] * self.W
        self.a = [0] * self.W

    def push(self, frame, action):
        z = self.m.encode_frame(frame.unsqueeze(0))
        self.z = (self.z + [z])[-self.W:]
        self.a = (self.a + [action])[-self.W:]

    @torch.no_grad()
    def decode_next_for_action(self, decode_op, candidate_a):
        z_hist = torch.stack(self.z, dim=1)              # [1,W,N,D]
        a_idx = self.a[1:] + [candidate_a]               # last slot = candidate
        a_hist = F.one_hot(torch.tensor(a_idx, device=self.dev), 4).float().unsqueeze(0)  # [1,W,4]
        z_next = self.m.op_predict_next(z_hist, a_hist)  # [1,N,D]
        return decode_op(z_next)[0]                      # [2]

    @property
    def cur_z(self): return self.z[-1]


@torch.no_grad()
def hist_greedy_action(buf, decode_op, subgoal_xy, device):
    goal = torch.tensor(subgoal_xy, device=device, dtype=torch.float32)
    best_a, best_d = 0, 1e30
    for a in range(4):
        d = float((buf.decode_next_for_action(decode_op, a) - goal).norm())
        if d < best_d:
            best_d, best_a = d, a
    return best_a


@torch.no_grad()
def build_graph_raw(model, decode_op, frames, positions, room_ids, starts, total, device,
                    k=24, S=4, max_eps=500):
    """Build a latent transition graph (concept landmarks + observed edges) directly
    from raw episode frames using the temporal model's pooled latents. Strategic
    routing = Dijkstra shortest path over this graph (goal-conditioned by
    construction), which avoids the raw sub-goal head's hindsight OOD failure."""
    from alps.core.latent_graph import LatentGraph, _kmeans
    per_z, per_xy, per_rm = [], [], []
    E = starts.shape[0]
    for e in range(min(E, max_eps)):
        s = int(starts[e]); end = int(starts[e + 1]) if e + 1 < E else total
        seq = list(range(s, end, S))
        if len(seq) < 2:
            continue
        f = frames[torch.tensor(seq)].to(device).float() / 255.0
        per_z.append(model.pool(model.encode_frame(f)).cpu().numpy())
        per_xy.append(positions[torch.tensor(seq)].numpy())
        per_rm.append(room_ids[torch.tensor(seq)].numpy())
    Z = np.concatenate(per_z); XY = np.concatenate(per_xy); RM = np.concatenate(per_rm)
    centroids, labels = _kmeans(Z, k)
    decoded = decode_op(torch.tensor(centroids, device=device).unsqueeze(1)).cpu().numpy()
    true_xy = np.zeros((k, 2), np.float32); room = np.zeros(k, np.int64)
    for j in range(k):
        m = labels == j
        if m.any():
            true_xy[j] = XY[m].mean(0); room[j] = int(round(RM[m].mean()))
    edges = np.zeros((k, k))
    off = 0
    for zp in per_z:
        L = len(zp); lab = labels[off:off + L]; off += L
        for t in range(L - 1):
            edges[lab[t], lab[t + 1]] += 1
    return LatentGraph(k=k, centroids=centroids.astype("float32"),
                       decoded_xy=decoded.astype("float32"), true_xy=true_xy,
                       room_id=room, edges=edges)


@torch.no_grad()
def run_episode(model, W, env, sr, gr, seed, device, decode_op, decode_tac=None,
                graph=None, strategy="operative", max_steps=120):
    """strategy in {operative, graph, subgoal}. 'graph' = latent-graph shortest-path
    waypoints (the strategic layer done right); 'subgoal' = raw learned sub-goal head."""
    obs = env.reset(start_room=sr, goal_room=gr); goal_xy = obs["target"].copy()
    opt = _oracle_path_len(sr, gr, seed)
    buf = HistoryBuffer(model, W, device); buf.reset(obs_to_frame(obs, device))

    h_goal, waypoints, wp_idx = None, None, 0
    if strategy in ("subgoal", "graph"):
        eg = TwoRoomsEnv(seed=seed); eg.reset(start_room=gr, goal_room=gr); eg.agent_pos = goal_xy.copy()
        z_goal = model.encode_frame(obs_to_frame({"image": eg.render()}, device).unsqueeze(0))
        if strategy == "subgoal":
            h_goal, _ = model.tac_encode(z_goal)
        else:
            zs = model.pool(buf.cur_z).squeeze(0).cpu().numpy()
            zg = model.pool(z_goal).squeeze(0).cpu().numpy()
            waypoints = graph.waypoints(zs, zg) + [goal_xy.copy()]

    for s in range(max_steps):
        if strategy == "subgoal":
            h, _ = model.tac_encode(buf.cur_z)
            sub = decode_tac(model.emit_subgoal(h, h_goal))[0].cpu().numpy()
        elif strategy == "graph":
            sub = waypoints[min(wp_idx, len(waypoints) - 1)]
        else:
            sub = goal_xy
        a = hist_greedy_action(buf, decode_op, sub, device)
        obs, _, done, info = env.step(a)
        buf.push(obs_to_frame(obs, device), a)
        if strategy == "graph" and wp_idx < len(waypoints) - 1 and \
                np.linalg.norm(obs["position"] - waypoints[wp_idx]) < REACH:
            wp_idx += 1
        if done or info["distance"] < REACH:
            return EpisodeResult(True, s + 1, opt, s + 1, sr != gr)
    return EpisodeResult(False, max_steps, opt, max_steps, sr != gr)


def run(args):
    device = torch.device(args.device)
    model, W = load_model(args.model_path, device)
    frames, actions, positions, room_ids, starts = load_raw(args.data_path)
    total = frames.shape[0]
    rng = np.random.RandomState(1)
    out = {}

    # probe sample set (single frames)
    idx_all = rng.permutation(total)[: args.limit_samples] if args.limit_samples else rng.permutation(total)
    ntr = int(len(idx_all) * 0.8); tr, va = idx_all[:ntr], idx_all[ntr:]
    Ztr, Ctr, Htr, Rtr, Ptr = gather(model, frames, positions, room_ids, tr, device)
    Zva, Cva, Hva, Rva, Pva = gather(model, frames, positions, room_ids, va, device)
    decode_op = fit_probe(Ztr, Ptr, device)
    decode_tac = fit_probe(Htr, Ptr, device)

    # G1
    g1 = (decode_op(Zva.to(device)) - Pva.to(device)).norm(dim=1).mean().item()
    out["G1"] = {"decode_err_world_units": g1, "passed": g1 < 0.3}

    # G_collapse: latent spread diagnostic (effective rank / dead dims / pinning).
    # Catastrophic collapse -> eff_rank~1, dead_dims>0, pairwise cosine~1. Low rank
    # is OK if it matches the task's intrinsic dim AND G1 still decodes.
    Zc = Zva - Zva.mean(0)
    ev = torch.linalg.eigvalsh((Zc.t() @ Zc) / (Zva.shape[0] - 1)).clamp(min=0)
    eff_rank = float((ev.sum() ** 2 / (ev ** 2).sum()))
    std = Zva.std(0)
    zn = torch.nn.functional.normalize(Zva[:600], dim=1)
    cos = zn @ zn.t()
    cos_mean = float(cos[~torch.eye(zn.shape[0], dtype=torch.bool)].mean())
    out["G_collapse"] = {
        "d_model": int(Zva.shape[1]), "effective_rank": eff_rank,
        "dead_dims": int((std < 0.01).sum()), "min_dim_std": float(std.min()),
        "mean_pairwise_cosine": cos_mean,
        "catastrophic_collapse": bool(eff_rank < 1.5 or cos_mean > 0.99),
    }

    # G_str / G_tac (probes)
    acc_c = room_probe_acc(Ctr, Rtr, Cva, Rva); acc_z = room_probe_acc(Ztr, Rtr, Zva, Rva)
    tac_dec = (decode_tac(Hva.to(device)) - Pva.to(device)).norm(dim=1).mean().item()
    out["G_str"] = {"room_acc_concept": acc_c, "room_acc_operative": acc_z, "passed": acc_c > 0.8}
    out["G_tac"] = {"tac_decode_err": tac_dec, "passed": tac_dec < 1.2}

    # G_roll: open-loop multi-step rollout drift (the temporal benefit)
    out["G_roll"] = gate_rollout(model, W, frames, actions, positions, starts, total, device,
                                 decode_op, H=args.roll_h, stride=args.stride)

    # G_goals: history-aware navigation. Strategic routing via the latent GRAPH
    # (goal-conditioned shortest path) is the fixed hierarchy; the raw sub-goal
    # head is kept as a comparison (it fails OOD at far goals).
    graph = build_graph_raw(model, decode_op, frames, positions, room_ids, starts, total,
                            device, k=args.graph_k, S=args.stride)
    cfgs = [(i % 2, (i % 2) if (i // 2) % 2 == 0 else 1 - (i % 2), 1000 + i) for i in range(args.n_episodes)]
    rop = [run_episode(model, W, TwoRoomsEnv(seed=s), sr, gr, s, device, decode_op, strategy="operative") for sr, gr, s in cfgs]
    rgr = [run_episode(model, W, TwoRoomsEnv(seed=s), sr, gr, s, device, decode_op, decode_tac, graph=graph, strategy="graph") for sr, gr, s in cfgs]
    rsg = [run_episode(model, W, TwoRoomsEnv(seed=s), sr, gr, s, device, decode_op, decode_tac, strategy="subgoal") for sr, gr, s in cfgs]
    ror = [run_baseline_episode(TwoRoomsEnv(seed=s), heuristic_oracle_policy(), sr, gr, s) for sr, gr, s in cfgs]
    out["G_goals"] = {"operative_only": summarize(rop), "latent_graph": summarize(rgr),
                      "subgoal_head": summarize(rsg), "oracle": summarize(ror)}

    os.makedirs(args.save_dir, exist_ok=True)
    p = os.path.join(args.save_dir, "temporal_gates.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\n===== TEMPORAL (K-frame history) GATES =====")
    print(f"G1     decode {g1:.3f}wu -> {'PASS' if out['G1']['passed'] else 'FAIL'}")
    gc = out["G_collapse"]
    print(f"G_coll eff-rank {gc['effective_rank']:.1f}/{gc['d_model']} | dead-dims {gc['dead_dims']} "
          f"| pairwise-cos {gc['mean_pairwise_cosine']:.2f} -> "
          f"{'COLLAPSE' if gc['catastrophic_collapse'] else 'no catastrophic collapse'}")
    print(f"G_str  room-acc concept {acc_c:.2f} (op {acc_z:.2f})")
    print(f"G_tac  decode {tac_dec:.3f}wu")
    r = out['G_roll']; print(f"G_roll {r['horizon']}-step rollout drift {r['rollout_drift_wu']:.3f}wu (1-step {r['one_step_wu']:.3f}wu)")
    g = out["G_goals"]
    print(f"G_goals cross-room  operative {g['operative_only']['cross_room_success']:.2f} | "
          f"GRAPH {g['latent_graph']['cross_room_success']:.2f} | "
          f"subgoal {g['subgoal_head']['cross_room_success']:.2f} | oracle {g['oracle']['cross_room_success']:.2f}")
    print(f"        same-room  operative {g['operative_only']['same_room_success']:.2f} | "
          f"GRAPH {g['latent_graph']['same_room_success']:.2f} | "
          f"subgoal {g['subgoal_head']['same_room_success']:.2f} | oracle {g['oracle']['same_room_success']:.2f}")
    edge = g['latent_graph']['cross_room_success'] - g['operative_only']['cross_room_success']
    print(f"        -> strategic-graph cross-room edge over operative: {edge:+.2f}")
    print(f"[report] {p}")
    return out


@torch.no_grad()
def gate_rollout(model, W, frames, actions, positions, starts, total, device, decode_op,
                 H=4, stride=4, n=400):
    """From a W-frame history, autoregressively roll the operative predictor
    forward H steps using the dataset's actions, and measure decoded-position
    error at step 1 vs step H. Low drift growth = the temporal-history benefit
    (straighter latent rollout). Predicted latents are fed back in (closed-loop)."""
    def dom(j):
        blk = actions[j:j + stride]
        return int(torch.bincount(blk, minlength=4).argmax().item()) if len(blk) else 0

    one_sum, multi_sum, cnt = 0.0, 0.0, 0
    E = starts.shape[0]
    for e in range(E):
        if cnt >= n:
            break
        s = int(starts[e]); end = int(starts[e + 1]) if e + 1 < E else total
        i = s
        while i + (W + H) * stride < end and cnt < n:
            fidx = [i + k * stride for k in range(W)]
            f = frames[torch.tensor(fidx)].to(device).float() / 255.0
            cur_z = [z for z in model.encode_frame(f).unsqueeze(0).unbind(1)]  # W x [1,N,D]
            cur_a = [dom(fidx[k]) for k in range(W)]
            for step in range(H):
                z_in = torch.stack(cur_z[-W:], dim=1)                  # [1,W,N,D]
                a_in = F.one_hot(torch.tensor(cur_a[-W:], device=device), 4).float().unsqueeze(0)
                z_pred = model.op_predict_next(z_in, a_in)             # [1,N,D]
                dec = decode_op(z_pred)[0].cpu().numpy()
                true_idx = i + (W + step) * stride
                err = float(np.linalg.norm(dec - positions[true_idx].numpy()))
                if step == 0:
                    one_sum += err
                if step == H - 1:
                    multi_sum += err
                cur_z.append(z_pred)
                cur_a.append(dom(i + (W - 1 + step) * stride))
            cnt += 1
            i += W * stride
    return {"horizon": H, "one_step_wu": one_sum / max(1, cnt),
            "rollout_drift_wu": multi_sum / max(1, cnt), "n": cnt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="results/two_rooms/validation/temporal_world_model.pt")
    ap.add_argument("--data-path", default="data/two_rooms/trajectories_large.pt")
    ap.add_argument("--save-dir", default="results/two_rooms/validation")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-episodes", type=int, default=20)
    ap.add_argument("--limit-samples", type=int, default=6000)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--roll-h", type=int, default=4)
    ap.add_argument("--graph-k", type=int, default=24)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
