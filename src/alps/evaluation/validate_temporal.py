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
        max_frames=ck.get("window", 6) + 1,
        use_projection_head=ck.get("use_projection_head", True)).to(device)
    msd = m.state_dict()
    sd = {k: v for k, v in ck["model_state_dict"].items() if k in msd and msd[k].shape == v.shape}
    m.load_state_dict(sd, strict=False); m.eval()
    return m, ck.get("window", 6)


def load_has_keys(path):
    """Load the per-frame has_key labels (complex datasets only); None if absent."""
    d = torch.load(path, map_location="cpu", weights_only=True)
    return d.get("has_keys", None)


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

    @torch.no_grad()
    def pooled_next_for_action(self, candidate_a):
        """Pooled PREDICTED next latent for an action (LABEL-FREE control: no decoder)."""
        z_hist = torch.stack(self.z, dim=1)
        a_idx = self.a[1:] + [candidate_a]
        a_hist = F.one_hot(torch.tensor(a_idx, device=self.dev), 4).float().unsqueeze(0)
        return self.m.pool(self.m.op_predict_next(z_hist, a_hist)).squeeze(0)   # [D]

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
def hist_greedy_action_latent(buf, sub_latent, device):
    """LABEL-FREE control: pick the action whose PREDICTED next pooled latent is nearest
    the sub-goal LATENT. No position decoder anywhere in the loop (LeWM-style)."""
    best_a, best_d = 0, 1e30
    for a in range(4):
        d = float((buf.pooled_next_for_action(a) - sub_latent).norm())
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
def gate_complex(model, W, graph, decode_op, device, n_episodes=30, max_steps=200):
    """COMPLEX mode (4-room, key->door->goal). Success requires reaching the goal
    WITH the key (env.done already enforces this). Compares operative-only vs
    latent-graph routing vs the complex heuristic oracle. This is the test the
    hierarchy must win — greedy alone cannot represent 'fetch key, then goal'."""
    from alps.benchmarks.two_rooms.data_generator import HeuristicPolicy

    def run(strategy, seed):
        env = TwoRoomsEnv(seed=seed, complex_mode=True, hazards=False); obs = env.reset()
        goal_xy = obs["target"].copy()
        buf = HistoryBuffer(model, W, device); buf.reset(obs_to_frame(obs, device))
        waypoints, wp = None, 0
        if strategy == "graph":
            eg = TwoRoomsEnv(seed=seed, complex_mode=True, hazards=False); eg.reset()
            eg.agent_pos = goal_xy.copy(); eg.has_key = True
            zs = model.pool(buf.cur_z).squeeze(0).cpu().numpy()
            zg = model.pool(model.encode_frame(obs_to_frame({"image": eg.render()}, device).unsqueeze(0))).squeeze(0).cpu().numpy()
            waypoints = graph.waypoints(zs, zg) + [goal_xy.copy()]
        for s in range(max_steps):
            sub = waypoints[min(wp, len(waypoints) - 1)] if strategy == "graph" else goal_xy
            a = hist_greedy_action(buf, decode_op, sub, device)
            obs, _, done, info = env.step(a); buf.push(obs_to_frame(obs, device), a)
            if strategy == "graph" and wp < len(waypoints) - 1 and \
                    np.linalg.norm(obs["position"] - waypoints[wp]) < REACH:
                wp += 1
            if done:
                return True
        return False

    def run_oracle(seed):
        from alps.benchmarks.two_rooms.optimal_planner import optimal_episode
        ok, _, _ = optimal_episode(seed, complex_mode=True, hazards=False, max_steps=max_steps)
        return ok

    res = {"operative": [], "graph": [], "oracle": []}
    for i in range(n_episodes):
        seed = 2000 + i
        res["operative"].append(run("operative", seed))
        res["graph"].append(run("graph", seed))
        res["oracle"].append(run_oracle(seed))
    return {k: float(np.mean(v)) for k, v in res.items()}


# ════════ FOUR-BRAIN controller (frozen latent space) ════════
@torch.no_grad()
def compute_node_concepts(model, graph, device):
    """Strategic concept (VQ) of each coarse graph node's mean LATENT centroid ->
    used to condition the tactical layer toward the next landmark."""
    cz = graph.z_centroids if graph.z_centroids is not None else graph.centroids
    return model.str_encode(torch.tensor(cz, device=device))[0]   # [k, D] strategic concepts


def fit_key_probe(Z, K, device, iters=300):
    """Frozen linear has-key probe (logistic). Returns w [D+1]; score=sigmoid([z,1]·w)."""
    X = torch.cat([Z.to(device), torch.ones(len(Z), 1, device=device)], 1)
    y = K.to(device).float()
    w = torch.zeros(X.shape[1], device=device, requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.05)
    for _ in range(iters):
        opt.zero_grad(); F.binary_cross_entropy_with_logits(X @ w, y).backward(); opt.step()
    return w.detach()


def make_featurize(decode_op, key_w, key_scale, device):
    """Map a pooled latent -> SEMANTIC feature [x, y] (or [x, y, key_scale·P(key)]).
    This is the space the strategic landmarks live in (frozen probes only)."""
    @torch.no_grad()
    def f(z_pooled_np):
        zt = torch.tensor(z_pooled_np, device=device, dtype=torch.float32).unsqueeze(0)  # [1,D]
        xy = decode_op(zt)[0].cpu().numpy()
        if key_w is None:
            return xy
        ks = torch.sigmoid((torch.cat([zt[0], torch.ones(1, device=device)]) * key_w).sum()).item()
        return np.concatenate([xy, [key_scale * ks]])
    return f


@torch.no_grad()
def build_graph_semantic(model, decode_op, key_w, key_scale, frames, positions, room_ids,
                         starts, total, device, k=8, S=2, max_eps=500, has_keys=None):
    """COARSE strategic graph whose landmarks live in DECODED (x, y[, has_key]) space
    rather than raw 128-d latent space. Raw-latent k-means is NOT position-faithful
    (non-positional variance dominates -> far positions and key states collapse into
    one node, so no routing happens). Clustering the frozen DECODED coordinates gives
    true room / key-state landmarks; reachability comes from the observed transition
    edges (a locked door simply has no no-key->goal-room edge).

    KEY landmark: the key spot is visited only briefly, so k-means never dedicates a
    centroid to it -> every controller misses the pickup. We therefore add an EXPLICIT
    landmark at the has_key 0->1 transition frames (the user's "keys" strategic goal),
    so the plan routes start -> KEY -> goal and a waypoint sits on the key. Edges are
    built at a small stride so the pickup transition is captured locally (large stride
    creates spurious unkeyed->keyed-goal shortcut edges that skip the key)."""
    from alps.core.latent_graph import LatentGraph, _kmeans
    per_z, per_xy, per_rm, per_hk, per_pick = [], [], [], [], []
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
        if has_keys is not None:
            hkseq = has_keys[torch.tensor(seq)].numpy()
            per_hk.append(hkseq)
            # pickup frames: last-unkeyed and first-keyed around a 0->1 flip
            pick = np.zeros(len(seq), bool)
            fl = np.where((hkseq[:-1] < 0.5) & (hkseq[1:] >= 0.5))[0]
            for t in fl:
                pick[t] = True; pick[t + 1] = True
            per_pick.append(pick)
    Z = np.concatenate(per_z); XY = np.concatenate(per_xy); RM = np.concatenate(per_rm)
    Zt = torch.tensor(Z, device=device)
    dec = decode_op(Zt).cpu().numpy()                                   # [M,2] decoded xy
    if key_w is None:
        feat, keyscore = dec, np.zeros(len(Z), np.float32)
    else:
        ks = torch.sigmoid(torch.cat([Zt, torch.ones(len(Zt), 1, device=device)], 1) @ key_w).cpu().numpy()
        feat = np.concatenate([dec, (key_scale * ks)[:, None]], 1).astype(np.float32)
        keyscore = ks.astype(np.float32)
    centroids_feat, labels = _kmeans(feat, k)
    # explicit KEY landmark = node index k (reassign pickup frames to it)
    kn = k
    if per_pick:
        pickmask = np.concatenate(per_pick)
        if pickmask.any():
            labels = labels.copy(); labels[pickmask] = kn; k = k + 1
    D = Z.shape[1]
    z_cent = np.zeros((k, D), np.float32); dxy = np.zeros((k, 2), np.float32)
    keyst = np.zeros(k, np.float32); true_xy = np.zeros((k, 2), np.float32); room = np.zeros(k, np.int64)
    for j in range(k):
        mj = labels == j
        if mj.any():
            z_cent[j] = Z[mj].mean(0); dxy[j] = dec[mj].mean(0); keyst[j] = keyscore[mj].mean()
            true_xy[j] = XY[mj].mean(0); room[j] = int(round(RM[mj].mean()))
    if k > kn:  # extend feature centroids with the explicit key node (index kn)
        kf = np.concatenate([dxy[kn], [key_scale * keyst[kn]]]).astype(np.float32)
        centroids_feat = np.vstack([centroids_feat, kf[None, :]])
    edges = np.zeros((k, k)); off = 0
    for zp in per_z:
        L = len(zp); lab = labels[off:off + L]; off += L
        for t in range(L - 1):
            edges[lab[t], lab[t + 1]] += 1
    g = LatentGraph(k=k, centroids=centroids_feat.astype("float32"), decoded_xy=dxy,
                    true_xy=true_xy, room_id=room, edges=edges,
                    z_centroids=z_cent, key_state=keyst,
                    key_node=(kn if k > kn else None))
    return g


@torch.no_grad()
def run_episode_4b(model, W, seed, sr, gr, device, decode_op, graph, featurize,
                   strategy, complex_mode=False, max_steps=140):
    """LABEL-FREE control (Option B): planning + acting entirely in LATENT space, no
    position decoder in the loop. strategy in {operative, graph}:
      operative : 1-step greedy toward the GOAL LATENT (System 1). Stalls at the wall /
                  locked door -- can't represent 'fetch key, then goal'.
      graph     : follow the latent-graph node CENTROIDS (latent) along the start->goal
                  shortest path, then the goal latent; advance a waypoint on entering its
                  Voronoi cell (nearest centroid). The latent transition graph routes
                  through key-acquisition on its own (keyed/unkeyed are distinct latent
                  nodes connected only by real pickup transitions) -- no key labels.
    `decode_op` is accepted for signature compatibility but NOT used for control."""
    env = TwoRoomsEnv(seed=seed, complex_mode=complex_mode, hazards=False)
    obs = env.reset() if complex_mode else env.reset(start_room=sr, goal_room=gr)
    goal_xy = obs["target"].copy()
    is_cross = True if complex_mode else (sr != gr)
    opt = max_steps if complex_mode else _oracle_path_len(sr, gr, seed)
    buf = HistoryBuffer(model, W, device); buf.reset(obs_to_frame(obs, device))

    # GOAL is specified as a goal IMAGE -> goal LATENT (no labels).
    eg = TwoRoomsEnv(seed=seed, complex_mode=complex_mode, hazards=False)
    eg.reset() if complex_mode else eg.reset(start_room=gr, goal_room=gr)
    eg.agent_pos = goal_xy.copy()
    if complex_mode:
        eg.has_key = True
    goal_lat = model.pool(model.encode_frame(obs_to_frame({"image": eg.render()}, device).unsqueeze(0))).squeeze(0)

    seg_nodes, lat_waypoints, wp, cents = [], None, 0, None
    if strategy == "graph":
        cents = torch.tensor(graph.centroids, device=device, dtype=torch.float32)  # [k,D] latent
        zs = model.pool(buf.cur_z).squeeze(0).cpu().numpy()
        sn = graph.node_of_latent(zs); gn = graph.node_of_latent(goal_lat.cpu().numpy())
        path = graph.shortest_path(sn, gn) or [gn]
        seg_nodes = path[1:] if len(path) > 1 else path[:]
        lat_waypoints = [cents[n] for n in seg_nodes] + [goal_lat]

    for s in range(max_steps):
        sub_lat = goal_lat if strategy == "operative" else lat_waypoints[min(wp, len(lat_waypoints) - 1)]
        a = hist_greedy_action_latent(buf, sub_lat, device)
        obs, _, done, info = env.step(a); buf.push(obs_to_frame(obs, device), a)
        # advance on Voronoi-cell entry in LATENT space (nearest centroid == target node)
        if seg_nodes and wp < len(seg_nodes):
            cur = model.pool(buf.cur_z).squeeze(0)
            if int((cents - cur).norm(dim=1).argmin()) == seg_nodes[wp]:
                wp += 1
        reached = done if complex_mode else (done or info["distance"] < REACH)  # ENV oracle = task metric
        if reached:
            return EpisodeResult(True, s + 1, opt, s + 1, is_cross)
    return EpisodeResult(False, max_steps, opt, max_steps, is_cross)


def _bfs_oracle_cfg(sr, gr, seed, complex_mode, max_steps=250):
    """BFS-optimal success on the EXACT (sr, gr, seed) config (matches the ablation)."""
    from alps.benchmarks.two_rooms.optimal_planner import bfs_actions
    pl = TwoRoomsEnv(seed=seed, complex_mode=complex_mode, hazards=False)
    pl.reset() if complex_mode else pl.reset(start_room=sr, goal_room=gr)
    plan = bfs_actions(pl)
    if plan is None:
        return False
    env = TwoRoomsEnv(seed=seed, complex_mode=complex_mode, hazards=False)
    env.reset() if complex_mode else env.reset(start_room=sr, goal_room=gr)
    for a in plan[:max_steps]:
        _, _, done, info = env.step(int(a))
        if done if complex_mode else (done or info["distance"] < REACH):
            return True
    return False


def gate_four_brain(model, W, coarse_graph, fine_graph, featurize, decode_op, device,
                    n_episodes, complex_mode=False):
    """3-tier ablation + BFS oracle (success rates). The tiers are control at three
    abstraction scales on the FROZEN latent space:
      operative : 1-step greedy to goal (no plan).
      strategic : COARSE latent-graph waypoints (few room/key landmarks).
      tactical  : FINE latent-graph waypoints (reachable sub-regions)."""
    if complex_mode:
        cfgs = [(0, 3, 2000 + i) for i in range(n_episodes)]
    else:
        cfgs = [(i % 2, (i % 2) if (i // 2) % 2 == 0 else 1 - (i % 2), 1000 + i) for i in range(n_episodes)]
    plan = {"operative": (None, "operative"), "strategic": (coarse_graph, "graph"),
            "tactical": (fine_graph, "graph")}
    out = {}
    for name, (gph, strat) in plan.items():
        res = [run_episode_4b(model, W, seed, sr, gr, device, decode_op, gph, featurize,
                              strat, complex_mode) for sr, gr, seed in cfgs]
        out[name] = summarize(res)
    out["oracle"] = float(np.mean([_bfs_oracle_cfg(sr, gr, seed, complex_mode) for sr, gr, seed in cfgs]))
    return out


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

    # has_key linear identifiability -- MEASUREMENT only (a frozen probe; never used by
    # the controller). Confirms the key state is in the latent (so the latent transition
    # graph can route through it WITHOUT key labels).
    if args.complex:
        hk = load_has_keys(args.data_path)
        if hk is not None:
            key_w = fit_key_probe(Ztr, hk[torch.from_numpy(tr)].float(), device)
            acc = ((torch.cat([Zva.to(device), torch.ones(len(Zva), 1, device=device)], 1) @ key_w > 0)
                   .float().cpu() == hk[torch.from_numpy(va)].float()).float().mean().item()
            out["G_key"] = {"has_key_probe_acc": acc, "passed": acc > 0.9}

    # H4 — label-free key detector (--h4-unsup-key)
    if getattr(args, "h4_unsup_key", False):
        print("--- [H4] running label-free key detector (surprise + VQ-flip) ---")
        pseudo_hk, surp, vq_fl = detect_key_pickups_unsup(
            model, frames, actions, starts, total, device, S=max(1, args.stride // 2))
        if args.complex:
            hk_gt = load_has_keys(args.data_path)
            if hk_gt is not None:
                out["G_H4"] = gate_h4_key_detector(pseudo_hk, hk_gt, starts, total,
                                                    S=max(1, args.stride // 2))
                print(f"G_H4  prec {out['G_H4']['precision']:.2f}  rec {out['G_H4']['recall']:.2f}  "
                      f"f1 {out['G_H4']['f1']:.2f}  -> {'PASS' if out['G_H4']['passed'] else 'FAIL'}")

    # ════ FOUR-BRAIN 3-tier ablation -- LABEL-FREE control (Option B) ════
    # Control graphs are PURE LATENT (raw k-means + observed transition edges); node
    # assignment is by latent distance (identity featurize); the controller plans and
    # acts entirely in latent space (no position decoder). strategic = COARSE graph
    # (few latent landmarks, stalls), tactical = FINE graph (reachable latent sub-regions,
    # threads). Small graph stride so the key-pickup transition is captured -> the latent
    # graph routes through key-acquisition on its own. Same in simple and complex.
    def build(k):
        return build_graph_raw(model, decode_op, frames, positions, room_ids, starts,
                               total, device, k=k, S=max(1, args.stride // 2))
    featurize = lambda z: z
    coarse_graph, fine_graph = build(args.coarse_k), build(args.fine_k)
    fb = gate_four_brain(model, W, coarse_graph, fine_graph, featurize, decode_op, device,
                         args.n_episodes, complex_mode=args.complex)
    out["G_4brain"] = fb

    # H3 — VQ concept-graph ablation (--vq-graph)
    if getattr(args, "vq_graph", False):
        print("--- [H3] building VQ concept-graph ---")
        vq_graph = build_graph_vq(model, decode_op, frames, positions, room_ids,
                                   starts, total, device, S=max(1, args.stride // 2))
        # Run the same 3-tier ablation but with VQ graph as the strategic tier
        fb_vq = gate_four_brain(model, W, vq_graph, fine_graph, featurize, decode_op,
                                device, args.n_episodes, complex_mode=args.complex)
        out["G_H3_vq_graph"] = {
            "operative": fb_vq["operative"],
            "vq_strategic": fb_vq["strategic"],
            "fine_tactical": fb_vq["tactical"],
            "oracle": fb_vq["oracle"],
            "vq_nodes_active": int((vq_graph.edges.sum(1) > 0).sum()),
        }
        metric = "overall_success" if args.complex else "cross_room_success"
        print(f"H3 VQ-graph cross-room: vq_strategic {fb_vq['strategic'][metric]:.2f}  "
              f"fine {fb_vq['tactical'][metric]:.2f}  "
              f"(ratio {fb_vq['strategic'][metric]/max(1e-9,fb_vq['tactical'][metric]):.2f}, "
              f"gate >=0.9)")

    # H2 — tactical emitter validity gates (--h2-emitter)
    if getattr(args, "h2_emitter", False):
        print("--- [H2] gating learned tactical subgoal emitter ---")
        h2 = gate_tactical_emitter(model, W, fine_graph, coarse_graph, decode_op,
                                    device, n_episodes=args.n_episodes,
                                    complex_mode=args.complex)
        out["G_H2_emitter"] = h2
        print(f"H2 emitter: coarse {h2['coarse_sr']:.2f} | emitter {h2['emitter_sr']:.2f} "
              f"| fine {h2['fine_sr']:.2f}  fallback {h2['fallback_rate']:.2f}  "
              f"-> {'PASS' if h2['passed_H2'] else 'FAIL'}")

    os.makedirs(args.save_dir, exist_ok=True)
    p = os.path.join(args.save_dir, "temporal_gates_complex.json" if args.complex else "temporal_gates.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=float)

    gc = out["G_collapse"]
    metric = "overall_success" if args.complex else "cross_room_success"
    tag = "COMPLEX key->door->goal" if args.complex else "simple"
    o, st, ta, orc = (fb['operative'][metric], fb['strategic'][metric],
                      fb['tactical'][metric], fb['oracle'])
    print(f"\n===== TEMPORAL FOUR-BRAIN GATES ({tag}) =====")
    print(f"G1     decode {g1:.3f}wu -> {'PASS' if out['G1']['passed'] else 'FAIL'}")
    print(f"G_coll eff-rank {gc['effective_rank']:.1f}/{gc['d_model']} | dead-dims {gc['dead_dims']} "
          f"| pairwise-cos {gc['mean_pairwise_cosine']:.2f} -> "
          f"{'COLLAPSE' if gc['catastrophic_collapse'] else 'no catastrophic collapse'}")
    print(f"G_str  room-acc concept {acc_c:.2f} (op {acc_z:.2f})")
    if "G_key" in out:
        print(f"G_key  has-key probe acc {out['G_key']['has_key_probe_acc']:.3f} "
              f"-> {'PASS' if out['G_key']['passed'] else 'FAIL'}")
    print(f"G_tac  decode {tac_dec:.3f}wu")
    r = out['G_roll']; print(f"G_roll {r['horizon']}-step drift {r['rollout_drift_wu']:.3f}wu (1-step {r['one_step_wu']:.3f}wu)")
    print(f"G_4brain {metric.replace('_success','')} success:")
    print(f"   operative(S1) {o:.2f}  ->  +strategic {st:.2f}  ->  +tactical(4-Brain) {ta:.2f}  |  oracle {orc:.2f}")
    print(f"   necessity:  strategic-over-operative {st-o:+.2f}  |  TACTICAL-over-strategic {ta-st:+.2f}")
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


# ══════════════════════════════════════════════════════════════════════════════
# H4 — Label-free key-pickup detector (WS-D)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def detect_key_pickups_unsup(model, frames, actions, starts, total, device,
                              surprise_q=0.90, vq_pos_eps=0.40, S=2, max_eps=500,
                              chunk=128):
    """H4: Discover key-pickup events WITHOUT labels (closes the has_keys supervision leak).

    Two label-free signals (BOTH on FROZEN model, no encoder update):
      1. SURPRISE   ||pool(op_predict_next(z_hist, a)) - pool(z_actual)||
                    spikes when the next frame is hard to predict (key vanishes from floor).
      2. VQ FLIP AT CONSTANT POSITION
                    strategic VQ code changes frame-to-frame while the decoded position
                    barely moves (state change without locomotion = pickup event).

    Returns
    -------
    pseudo_hk : float32 tensor [total]
        Per-frame pseudo-has_key score.  Within each episode, frames from the FIRST
        detected pickup event onward are set to 1.0; frames before to 0.0.
        Drop-in replacement for `has_keys` in build_graph_semantic / fourth_brain.
    surprise   : float32 array [total]  (for diagnostics / G_H4 gate)
    vq_flip    : bool array [total]     (for diagnostics)
    """
    # W = history window used at eval (= training window; stored in frame_emb shape - 1)
    W = int(model.op_predictor.frame_emb.shape[1]) - 1
    if W <= 0:
        W = max(1, int(model.op_predictor.frame_emb.shape[1]) - 1)

    # -- Step 1: encode all frames in batches -> pooled latent + VQ code -------
    all_z = torch.zeros(total, model.d_model, dtype=torch.float32)
    all_vq = torch.zeros(total, dtype=torch.long)
    for c0 in range(0, total, chunk):
        c1 = min(c0 + chunk, total)
        f = frames[c0:c1].to(device).float() / 255.0
        z = model.encode_frame(f)                     # [B,N,D]
        all_z[c0:c1] = model.pool(z).cpu()           # [B,D] pooled
        _, _, idx = model.str_encode(z)               # [B] VQ code indices
        all_vq[c0:c1] = idx.cpu()

    # -- Step 2: decode positions (for VQ-flip-at-constant-position check) -----
    all_pos = torch.zeros(total, 2, dtype=torch.float32)
    for c0 in range(0, total, chunk):
        c1 = min(c0 + chunk, total)
        z_chunk = all_z[c0:c1].to(device)
        all_pos[c0:c1] = model.decode_pos(z_chunk).cpu()

    # -- Step 3: per-frame operative surprise + VQ-flip -------------------------
    surprise = np.zeros(total, dtype=np.float32)
    vq_flip  = np.zeros(total, dtype=bool)
    E = starts.shape[0]

    for e in range(min(E, max_eps)):
        s = int(starts[e]); end = int(starts[e + 1]) if e + 1 < E else total
        seq = list(range(s, end, S))
        L = len(seq)
        if L < W + 1:
            continue

        # VQ-flip at constant position (no model call needed, uses precomputed)
        for ti in range(1, L):
            gi, gprev = seq[ti], seq[ti - 1]
            if int(all_vq[gi]) != int(all_vq[gprev]):
                pos_delta = float((all_pos[gi] - all_pos[gprev]).norm())
                if pos_delta < vq_pos_eps:
                    vq_flip[gi] = True

        # Operative surprise (needs history of W frames)
        f_seq = frames[torch.tensor(seq)].to(device).float() / 255.0
        z_seq = model.encode_frame(f_seq)              # [L,N,D]
        for ti in range(W, L):
            z_hist = z_seq[ti - W:ti].unsqueeze(0)    # [1,W,N,D]
            a_sub = [int(actions[seq[ti - W + k]]) for k in range(W)]
            a_hist = F.one_hot(torch.tensor(a_sub, device=device), 4).float().unsqueeze(0)
            z_pred = model.op_predict_next(z_hist, a_hist)           # [1,N,D]
            pred_p = model.pool(z_pred).squeeze(0)                   # [D]
            actual_p = model.pool(z_seq[ti:ti+1]).squeeze(0)         # [D]
            surprise[seq[ti]] = float((pred_p - actual_p).norm())

    # -- Step 4: threshold surprise at quantile --------------------------------
    nz = surprise[surprise > 0]
    thr = float(np.quantile(nz, surprise_q)) if len(nz) > 0 else float("inf")
    pickup_event = (surprise > thr) | vq_flip         # union of both signals

    # -- Step 5: build pseudo-has_key: first pickup → 1.0 for rest of episode --
    pseudo_hk = torch.zeros(total, dtype=torch.float32)
    for e in range(E):
        s = int(starts[e]); end = int(starts[e + 1]) if e + 1 < E else total
        seq = list(range(s, end, S))
        for ti, gi in enumerate(seq):
            if pickup_event[gi]:
                # all frames from this point to episode end = has_key
                for gj in range(gi, end):
                    pseudo_hk[gj] = 1.0
                break

    return pseudo_hk, surprise, vq_flip


@torch.no_grad()
def gate_h4_key_detector(pseudo_hk, true_hk, starts, total, S=2, max_eps=500):
    """H4 precision/recall gate — compare label-free pseudo_hk to ground-truth labels.
    Labels used ONLY for measurement (never for training or control).
    Returns precision, recall, f1 at the episode level (did we detect a pickup in
    every episode that has one?).
    """
    E = starts.shape[0]
    tp = fp = fn = 0
    for e in range(min(E, max_eps)):
        s = int(starts[e]); end = int(starts[e + 1]) if e + 1 < E else total
        seq = list(range(s, end, S))
        gt_has = any(float(true_hk[gi]) > 0.5 for gi in seq if gi < total)
        det_has = any(float(pseudo_hk[gi]) > 0.5 for gi in seq if gi < total)
        if gt_has and det_has: tp += 1
        elif not gt_has and det_has: fp += 1
        elif gt_has and not det_has: fn += 1
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"precision": prec, "recall": rec, "f1": f1, "passed": rec >= 0.8}


# ══════════════════════════════════════════════════════════════════════════════
# H3 — VQ concept-graph planning (WS-C)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def build_graph_vq(model, decode_op, frames, positions, room_ids, starts, total, device,
                   S=2, max_eps=500):
    """H3: Plan over LEARNED VQ code indices as discrete strategic nodes.

    Nodes  = model.vq.num_embeddings (64 by default) — the learned codebook entries.
    Edges  = empirical code-to-code transition counts from dataset episodes.
    Decoded position per node = frozen position probe decode of the VQ embedding.
    True position  = mean true (x,y) of frames assigned to each code (validation only).

    Advantage over k-means graph: nodes are the model's own discrete abstraction,
    so the plan is a human-readable sequence of learned strategic concepts
    (code 5 → code 12 → code 41 → …). Code→room purity is already measured by G_str.
    """
    from alps.core.latent_graph import LatentGraph
    num_codes = int(model.vq.num_embeddings)                # e.g. 64
    # VQ codebook embeddings — each is a [D]-dim latent representative
    code_embs = model.vq.embeddings.weight.detach().cpu()  # [K, D]
    decoded_xy = decode_op(code_embs.to(device)).cpu().numpy()  # [K, 2]

    edges = np.zeros((num_codes, num_codes), dtype=np.float64)
    sum_xy = np.zeros((num_codes, 2), dtype=np.float64)
    sum_rm = np.zeros(num_codes, dtype=np.float64)
    count  = np.zeros(num_codes, dtype=np.float64)

    E = starts.shape[0]
    for e in range(min(E, max_eps)):
        s = int(starts[e]); end = int(starts[e + 1]) if e + 1 < E else total
        seq = list(range(s, end, S))
        if len(seq) < 2:
            continue
        f = frames[torch.tensor(seq)].to(device).float() / 255.0
        z = model.encode_frame(f)
        _, _, idx = model.str_encode(z)    # [L] code indices in [0, K-1]
        codes = idx.cpu().numpy()
        xy = positions[torch.tensor(seq)].numpy()
        rm = room_ids[torch.tensor(seq)].numpy()
        for t in range(len(codes)):
            c = int(codes[t])
            sum_xy[c] += xy[t]; sum_rm[c] += rm[t]; count[c] += 1
        for t in range(len(codes) - 1):
            edges[int(codes[t]), int(codes[t + 1])] += 1

    true_xy = np.zeros((num_codes, 2), np.float32)
    room_id = np.zeros(num_codes, np.int64)
    for j in range(num_codes):
        if count[j] > 0:
            true_xy[j] = sum_xy[j] / count[j]
            room_id[j] = int(round(sum_rm[j] / count[j]))

    return LatentGraph(k=num_codes,
                       centroids=code_embs.numpy().astype(np.float32),
                       decoded_xy=decoded_xy.astype(np.float32),
                       true_xy=true_xy, room_id=room_id, edges=edges,
                       z_centroids=code_embs.numpy().astype(np.float32))


# ══════════════════════════════════════════════════════════════════════════════
# H2 — Learned tactical emitter with validity gates (WS-B)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def gate_tactical_emitter(model, W, fine_graph, coarse_graph, decode_op, device,
                           n_episodes=30, complex_mode=False, max_steps=140,
                           progress_eps=0.05, reachability_thresh=None):
    """H2: Validate the LEARNED tactical subgoal emitter vs fine-graph upper bound.

    Each step, the emitter proposes subgoal = decode(emit_subgoal(h_t, h_goal)).
    Two validity gates (BOTH must pass for the emitter proposal to be used):
      PROGRESS    : decoded proposal is strictly closer to the goal than current pos.
      REACHABILITY: proposal is within REACH of at least one fine-graph node that is
                    reachable from the current node (i.e., not behind a wall).
    If either gate fails → fall back to the fine-graph waypoint (log fallback).

    Returns dict with: success rates (coarse / coarse+emitter / fine), fallback_rate,
    emitter_pass_rate (fraction of steps where emitter proposal passes both gates).
    Gate: coarse+emitter >= 0.9 * fine AND fallback_rate < 0.30.
    """
    if reachability_thresh is None:
        reachability_thresh = REACH * 2.0   # decoded proposal within 2 * REACH of fine node

    fine_cents = torch.tensor(fine_graph.centroids, device=device, dtype=torch.float32)
    fine_decoded = fine_graph.decoded_xy    # [fine_k, 2]

    def cfgs():
        if complex_mode:
            return [(0, 3, 2000 + i) for i in range(n_episodes)]
        return [(i % 2, (i % 2) if (i // 2) % 2 == 0 else 1 - (i % 2), 1000 + i)
                for i in range(n_episodes)]

    def run_emitter(strategy, seed, sr, gr):
        """strategy in {coarse, emitter, fine}."""
        env = TwoRoomsEnv(seed=seed, complex_mode=complex_mode, hazards=False)
        obs = env.reset() if complex_mode else env.reset(start_room=sr, goal_room=gr)
        goal_xy = obs["target"].copy()
        buf = HistoryBuffer(model, W, device); buf.reset(obs_to_frame(obs, device))

        # Goal latent (label-free: render goal image)
        eg = TwoRoomsEnv(seed=seed, complex_mode=complex_mode, hazards=False)
        eg.reset() if complex_mode else eg.reset(start_room=gr, goal_room=gr)
        eg.agent_pos = goal_xy.copy()
        if complex_mode:
            eg.has_key = True
        goal_lat = model.pool(model.encode_frame(
            obs_to_frame({"image": eg.render()}, device).unsqueeze(0))).squeeze(0)
        h_goal, _ = model.tac_encode(
            model.encode_frame(obs_to_frame({"image": eg.render()}, device).unsqueeze(0)))
        h_goal = h_goal.squeeze(0)

        # Graph for coarse/emitter fallback
        gr_cents = torch.tensor(coarse_graph.centroids, device=device, dtype=torch.float32)
        zs = model.pool(buf.cur_z).squeeze(0).cpu().numpy()
        gn = coarse_graph.node_of_latent(goal_lat.cpu().numpy())
        sn = coarse_graph.node_of_latent(zs)
        path = coarse_graph.shortest_path(sn, gn) or [gn]
        seg_nodes = path[1:] if len(path) > 1 else path[:]
        wp = 0

        total_steps = 0; emit_pass = 0; emit_total = 0; fallback_count = 0
        for s in range(max_steps):
            cur_z = model.pool(buf.cur_z).squeeze(0)
            cur_pos_dec = decode_op(cur_z.unsqueeze(0))[0].cpu().numpy()
            goal_pos_dec = decode_op(goal_lat.unsqueeze(0))[0].cpu().numpy()

            if strategy == "fine":
                sub_lat = (fine_cents[seg_nodes[wp]] if wp < len(seg_nodes) else goal_lat)
                # recompute fine graph seg if needed
                if s == 0:
                    fsn = fine_graph.node_of_latent(cur_z.cpu().numpy())
                    fgn = fine_graph.node_of_latent(goal_lat.cpu().numpy())
                    fpath = fine_graph.shortest_path(fsn, fgn) or [fgn]
                    seg_nodes = fpath[1:] if len(fpath) > 1 else fpath[:]
                    wp = 0
                sub_lat = fine_cents[seg_nodes[min(wp, len(seg_nodes)-1)]] if seg_nodes else goal_lat
            elif strategy == "coarse":
                sub_lat = gr_cents[seg_nodes[min(wp, len(seg_nodes)-1)]] if seg_nodes else goal_lat
            else:  # emitter
                emit_total += 1
                h_t, _ = model.tac_encode(buf.cur_z)
                h_t = h_t.squeeze(0)
                prop_h = model.emit_subgoal(h_t.unsqueeze(0), h_goal.unsqueeze(0)).squeeze(0)
                prop_pos = decode_op(prop_h.unsqueeze(0))[0].cpu().numpy()
                goal_dist = np.linalg.norm(goal_pos_dec - cur_pos_dec)
                prop_dist = np.linalg.norm(goal_pos_dec - prop_pos)
                progress_ok = prop_dist < goal_dist - progress_eps
                reach_ok = np.min(np.linalg.norm(fine_decoded - prop_pos, axis=1)) < reachability_thresh
                if progress_ok and reach_ok:
                    emit_pass += 1
                    sub_lat = model.pool(model.encode_frame(
                        obs_to_frame({"image": env.render()}, device).unsqueeze(0))).squeeze(0)
                    # use tac_decoded proposal as xy subgoal via decode_op
                    sub_lat = prop_h
                else:
                    fallback_count += 1
                    sub_lat = gr_cents[seg_nodes[min(wp, len(seg_nodes)-1)]] if seg_nodes else goal_lat

            a = hist_greedy_action_latent(buf, sub_lat, device)
            obs, _, done, info = env.step(a); buf.push(obs_to_frame(obs, device), a)
            # advance coarse waypoint
            cur_n = int((gr_cents - model.pool(buf.cur_z).squeeze(0)).norm(dim=1).argmin())
            if seg_nodes and wp < len(seg_nodes) and cur_n == seg_nodes[wp]:
                wp += 1
            total_steps += 1
            if done if complex_mode else (done or info.get("distance", 1) < REACH):
                return True, emit_pass, emit_total, fallback_count
        return False, emit_pass, emit_total, fallback_count

    results = {"coarse": [], "emitter": [], "fine": []}
    emit_pass_total = emit_total_all = fallback_all = 0
    for sr, gr, seed in cfgs():
        for strat in ("coarse", "emitter", "fine"):
            ok, ep, et, fb = run_emitter(strat, seed, sr, gr)
            results[strat].append(ok)
            if strat == "emitter":
                emit_pass_total += ep; emit_total_all += et; fallback_all += fb

    coarse_sr = float(np.mean(results["coarse"]))
    emitter_sr = float(np.mean(results["emitter"]))
    fine_sr = float(np.mean(results["fine"]))
    emit_pass_rate = emit_pass_total / max(1, emit_total_all)
    fallback_rate  = fallback_all / max(1, emit_total_all)
    passed = (emitter_sr >= 0.9 * fine_sr) and (fallback_rate < 0.30)
    return {"coarse_sr": coarse_sr, "emitter_sr": emitter_sr, "fine_sr": fine_sr,
            "emitter_pass_rate": emit_pass_rate, "fallback_rate": fallback_rate,
            "passed_H2": passed, "n_episodes": n_episodes}


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
    ap.add_argument("--coarse-k", type=int, default=6,
                    help="STRATEGIC (coarse) landmarks: few room/key-state nodes -> sparse "
                         "waypoints that stall when a landmark sits behind a wall.")
    ap.add_argument("--fine-k", type=int, default=24,
                    help="TACTICAL (fine) landmarks: reachable sub-regions that thread doors.")
    ap.add_argument("--complex", action="store_true", help="4-room key-gated complex mode")
    ap.add_argument("--h4-unsup-key", action="store_true",
                    help="H4: run label-free key detector and report G_H4 gate "
                         "(precision/recall vs ground-truth labels, measurement only).")
    ap.add_argument("--vq-graph", action="store_true",
                    help="H3: build and evaluate VQ concept-graph as an extra ablation "
                         "tier (plan over learned discrete codes).")
    ap.add_argument("--h2-emitter", action="store_true",
                    help="H2: gate the learned tactical subgoal emitter with validity "
                         "gates (progress + reachability) and log fallback rate.")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
