"""SEE the low-level control failure on the egocentric 2-rooms. For each seed, trace BOTH:
  operative (greedy to the decoded goal)  -- expected to stall at the wall (greedy can't route),
  strategic (follow graph waypoints)       -- SHOULD thread the door; does the graph route there?
Plots true trajectories + the strategic waypoints over the wall/door, so we can see whether the
failure is routing (no door waypoint / path through wall) or control (can't thread the door).
Usage: python trace_control.py <data.pt> <model.pt> [--complex] [--k 24]
"""
import sys; sys.path.insert(0, "src")
import numpy as np, torch, argparse
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from alps.benchmarks.two_rooms.environment import TwoRoomsEnv
from alps.benchmarks.two_rooms.world_model_planning import obs_to_frame
from alps.training.train_hier import load_raw
from alps.evaluation.validate_temporal import (load_model, HistoryBuffer, fit_ridge_decode,
                                               calibrate_bn, gather_pred_grids, build_graph_raw, REACH)

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("model")
ap.add_argument("--complex", action="store_true"); ap.add_argument("--ctrl-k", type=int, default=5)
ap.add_argument("--grid", type=int, default=8); ap.add_argument("--n", type=int, default=6)
ap.add_argument("--k", type=int, default=24); ap.add_argument("--out", default="_trace.png")
a = ap.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.set_grad_enabled(False)
m, W = load_model(a.model, dev)
frames, actions, positions, room_ids, starts = load_raw(a.data)
tot = frames.shape[0]; calibrate_bn(m, frames, dev)
g = a.grid; readout = lambda z: m.spatial_readout(z, grid=g)

Zp, Yp = gather_pred_grids(m, frames, positions, actions, starts, tot, W, dev, n_win=3000)
Xp = torch.cat([readout(Zp[c:c + 128].to(dev)).detach().cpu().float() for c in range(0, len(Zp), 128)])
decode_calib = (lambda grid, _c=fit_ridge_decode(Xp, Yp, dev): _c(readout(grid)))   # A3 for imagination
rng = np.random.RandomState(0); idx = rng.permutation(tot)[:8000]
Xr = torch.cat([readout(m.encode_frame(frames[torch.as_tensor(idx[c:c+128])].to(dev).float()/255.)).cpu()
                for c in range(0, len(idx), 128)])
ridge = fit_ridge_decode(Xr, positions[torch.as_tensor(idx)], dev)   # readout-space -> position
decode_real = lambda grid: ridge(readout(grid))                      # frame token-grid -> position
# graph nodes live in readout space, so build_graph_raw gets the RAW ridge (not decode_real,
# which would re-apply the readout to already-readout-space centroids -> shape mismatch)
graph = build_graph_raw(m, ridge, frames, positions, room_ids, starts, tot, dev, k=a.k, S=2, readout=readout)
graph.decoded_xy = np.asarray(graph.decoded_xy).reshape(-1, 2)   # ridge keeps a singleton dim

@torch.no_grad()
def _setup(seed):
    env = TwoRoomsEnv(seed=seed, complex_mode=a.complex, hazards=False, egocentric=True)
    obs = env.reset() if a.complex else env.reset(start_room=0, goal_room=1)
    goal_xy = obs["target"].copy()
    buf = HistoryBuffer(m, W, dev, readout=readout); buf.reset(obs_to_frame(obs, dev))
    eg = TwoRoomsEnv(seed=seed, complex_mode=a.complex, hazards=False, egocentric=True)
    eg.reset() if a.complex else eg.reset(start_room=1, goal_room=1); eg.agent_pos = goal_xy.copy()
    if a.complex: eg.has_key = True
    goal_grid = m.encode_frame(obs_to_frame({"image": eg.render()}, dev).unsqueeze(0))
    return env, obs, buf, goal_xy, goal_grid

@torch.no_grad()
def trace(seed, strategy):
    env, obs, buf, goal_xy, goal_grid = _setup(seed)
    goal_pos = decode_real(goal_grid)[0].cpu().numpy()
    waypoints, wp = [goal_pos], 0
    if strategy == "graph":
        sn = graph.node_of_latent(readout(buf.cur_z).squeeze(0).cpu().numpy())
        gn = graph.node_of_latent(readout(goal_grid).squeeze(0).cpu().numpy())
        path = graph.shortest_path(sn, gn) or [gn]; seg = path[1:] if len(path) > 1 else path
        waypoints = [graph.decoded_xy[n] for n in seg] + [goal_pos]
    traj, solved = [obs["position"].copy()], False
    for s in range(140):
        sub = waypoints[min(wp, len(waypoints) - 1)]
        best_a, best_d = 0, 1e30
        for act in range(4):
            pa = buf.rollout_decode(decode_calib, act, a.ctrl_k).cpu().numpy()
            d = float(np.linalg.norm(pa - sub))
            if d < best_d: best_d, best_a = d, act
        obs, _, done, info = env.step(best_a); buf.push(obs_to_frame(obs, dev), best_a)
        traj.append(obs["position"].copy())
        if strategy == "graph" and wp < len(waypoints) - 1:
            if np.linalg.norm(decode_real(buf.cur_z)[0].cpu().numpy() - waypoints[wp]) < REACH: wp += 1
        if done or info.get("distance", 9) < REACH: solved = True; break
    return np.array(traj), np.array(waypoints), solved

fig, axes = plt.subplots(2, (a.n + 1) // 2, figsize=(4 * ((a.n + 1) // 2), 8))
for i, ax in enumerate(np.array(axes).flat[:a.n]):
    if not a.complex:
        ax.plot([5, 5], [0, 4.5], 'k', lw=5); ax.plot([5, 5], [5.5, 10], 'k', lw=5)
    to, _, so = trace(3000 + i, "operative")
    tg, wpts, sg = trace(3000 + i, "graph")
    ax.plot(to[:, 0], to[:, 1], '-', lw=1, color='tab:red', alpha=0.7, label=f'operative ({"solved" if so else "stuck"})')
    ax.plot(tg[:, 0], tg[:, 1], '-', lw=1.5, color='tab:blue', label=f'strategic ({"solved" if sg else "stuck"})')
    ax.plot(wpts[:, 0], wpts[:, 1], 'x', ms=9, color='orange', mew=2, label='waypoints')
    ax.scatter(*to[0], c='g', s=90, zorder=5); ax.scatter(*wpts[-1], c='r', s=150, marker='*', zorder=5)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect('equal')
    ax.set_title(f"seed{3000+i}", fontsize=9)
    if i == 0: ax.legend(fontsize=7, loc='upper left')
fig.suptitle("green=start  red*=goal  red=operative(greedy)  blue=strategic(waypoints)  orange x=waypoints")
fig.tight_layout(); fig.savefig(a.out, dpi=120); print(f"saved {a.out}")
