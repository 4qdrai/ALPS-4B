"""
ALPS-4B Two Rooms — ablation ladder (the core "edge" evidence).

Runs the identical navigation evaluation across a ladder of controllers and
reports success / SPL / compute for each. The architecture's edge is proven
ONLY if higher rungs beat lower ones on CROSS-ROOM success while keeping
compute (replans) reasonable:

  rung 0  random              : floor
  rung 2  operative greedy MPC: world model + decoder, sub-goal = goal only
                                (expected to stall at the wall on cross-room)
  rung 4a strategic [door,goal]: hand-specified waypoint (hierarchy, hard-coded)
  rung 4b latent-graph plan    : waypoints from the learned latent transition
                                 graph shortest path  <-- the real strategic layer
  rung 5  heuristic oracle     : near-optimal on true state (ceiling)

Outputs results/two_rooms/ablation/ladder_metrics.json + a bar chart + a
latent-graph figure.

USAGE
    PYTHONPATH=src python -m alps.benchmarks.two_rooms.run_ablation_ladder \
        --model-path results/two_rooms/validation/repr_world_model_fs4.pt \
        --n-episodes 40
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, "src")

import argparse, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alps.benchmarks.two_rooms.dataset import TwoRoomsDataset
from alps.benchmarks.two_rooms.environment import TwoRoomsEnv
from alps.benchmarks.two_rooms.world_model_planning import (
    WorldModelMPC, random_policy, heuristic_oracle_policy, strategic_waypoints,
    run_world_model_episode, run_baseline_episode, summarize, obs_to_frame,
    greedy_step_action, EpisodeResult, _oracle_path_len, DOOR_XY,
)
from alps.core.latent_graph import build_latent_graph
from alps.evaluation.repr_decoder_gate import ReprWorldModel, gate_g1, split_dataset


def load_world_model(path: str, device) -> ReprWorldModel:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    m = ReprWorldModel(d_model=ckpt.get("d_model", 128)).to(device)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m


def _episode_configs(n_episodes: int):
    """Balanced same-room / cross-room configs with deterministic seeds."""
    cfgs = []
    for i in range(n_episodes):
        start_room = i % 2
        goal_room = start_room if (i // 2) % 2 == 0 else 1 - start_room
        cfgs.append((start_room, goal_room, 1000 + i))
    return cfgs


def run_graph_episode(model, decode_fn, graph, env, device, start_room, goal_room, seed,
                      **kw) -> EpisodeResult:
    """MPC where waypoints come from the latent-graph shortest path."""
    obs = env.reset(start_room=start_room, goal_room=goal_room)
    start_xy, goal_xy = obs["position"].copy(), obs["target"].copy()
    is_cross = start_room != goal_room
    optimal_len = _oracle_path_len(start_room, goal_room, seed)

    # encode start & goal to pooled latents -> graph waypoints
    with torch.no_grad():
        z_start = model.encode_frame(obs_to_frame(obs, device).unsqueeze(0)).mean(1).squeeze(0).cpu().numpy()
        env_goal = TwoRoomsEnv(seed=seed); env_goal.reset(start_room=goal_room, goal_room=goal_room)
        env_goal.agent_pos = goal_xy.copy()
        g_img = obs_to_frame({"image": env_goal.render()}, device)
        z_goal = model.encode_frame(g_img.unsqueeze(0)).mean(1).squeeze(0).cpu().numpy()
    waypoints = graph.waypoints(z_start, z_goal)
    waypoints.append(goal_xy.copy())  # always end exactly at the goal

    max_steps = kw.get("max_steps", 150)
    reach_radius = kw.get("reach_radius", 0.6)

    wp_idx, steps, reached = 0, 0, False
    while steps < max_steps and not reached:
        sub = waypoints[min(wp_idx, len(waypoints) - 1)]
        a = greedy_step_action(model, decode_fn, obs_to_frame(obs, device), sub, device)
        obs, _, done, info = env.step(a)
        steps += 1
        if wp_idx < len(waypoints) - 1 and np.linalg.norm(obs["position"] - waypoints[wp_idx]) < reach_radius:
            wp_idx += 1
        if done or info["distance"] < reach_radius:
            reached = True
    return EpisodeResult(reached, steps, optimal_len, replans=steps, is_cross=is_cross)


def plot_ladder(metrics: dict, save_path: str):
    rungs = list(metrics.keys())
    cross = [metrics[r]["cross_room_success"] * 100 for r in rungs]
    same = [metrics[r]["same_room_success"] * 100 for r in rungs]
    x = np.arange(len(rungs))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, same, 0.4, label="same-room", color="#1f77b4")
    ax.bar(x + 0.2, cross, 0.4, label="cross-room", color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels(rungs, rotation=15, ha="right")
    ax.set_ylabel("success rate (%)"); ax.set_ylim(0, 105)
    ax.set_title("ALPS-4B Two Rooms — ablation ladder (hierarchy benefit)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for i, (s, c) in enumerate(zip(same, cross)):
        ax.text(i - 0.2, s + 1, f"{s:.0f}", ha="center", fontsize=8)
        ax.text(i + 0.2, c + 1, f"{c:.0f}", ha="center", fontsize=8)
    fig.tight_layout(); os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"[figure] {save_path}")


def plot_graph(graph, save_path: str):
    fig, ax = plt.subplots(figsize=(7, 7))
    n = graph.k
    row = graph.edges.sum(1, keepdims=True)
    p = graph.edges / np.clip(row, 1e-8, None)
    for i in range(n):
        for j in range(n):
            if p[i, j] > 0.05 and i != j:
                a, b = graph.decoded_xy[i], graph.decoded_xy[j]
                ax.plot([a[0], b[0]], [a[1], b[1]], color="#888", alpha=min(1, p[i, j]), lw=1)
    sc = ax.scatter(graph.decoded_xy[:, 0], graph.decoded_xy[:, 1],
                    c=graph.room_id, cmap="coolwarm", s=120, edgecolors="k", zorder=3)
    ax.axvline(5.0, color="brown", lw=2, alpha=0.5)
    ax.add_patch(plt.Rectangle((4.9, 4.5), 0.2, 1.0, color="white", zorder=2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal")
    ax.set_title("Latent transition graph (nodes=landmarks, decoded positions)")
    fig.tight_layout(); os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"[figure] {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="results/two_rooms/validation/repr_world_model_fs4.pt")
    ap.add_argument("--data-path", default="data/two_rooms/trajectories.pt")
    ap.add_argument("--save-dir", default="results/two_rooms/ablation")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--frame-skip", type=int, default=4)
    ap.add_argument("--n-episodes", type=int, default=40)
    ap.add_argument("--graph-k", type=int, default=16)
    ap.add_argument("--limit-clips", type=int, default=0)
    ap.add_argument("--rungs", default="0,2,4a,4b,5",
                    help="comma list subset of {0,2,4a,4b,5}")
    # CEM/MPC knobs (lower for fast local runs, raise on the A40)
    ap.add_argument("--n-cand", type=int, default=100)
    ap.add_argument("--n-iter", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--replan-interval", type=int, default=4)
    args = ap.parse_args()
    cem = dict(horizon=args.horizon, n_cand=args.n_cand,
               n_elite=max(1, args.n_cand // 10), n_iter=args.n_iter,
               replan_interval=args.replan_interval, max_steps=args.max_steps)

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    dataset = TwoRoomsDataset(args.data_path, clip_length=8, stride=4, frame_skip=args.frame_skip)
    if args.limit_clips:
        dataset.clip_indices = dataset.clip_indices[: args.limit_clips]
    train_idx, val_idx = split_dataset(dataset, val_frac=0.2, seed=0)

    print(f"[load] world model <- {args.model_path}")
    model = load_world_model(args.model_path, device)

    # decoder probe (also yields the G1 number for the report)
    g1, decode_fn = gate_g1(model, dataset, train_idx, val_idx, device, probe_epochs=80)
    print(f"[decoder] held-out error = {g1['held_out_decode_error_world_units']:.3f} world units")

    graph = build_latent_graph(model, decode_fn, dataset, device, k=args.graph_k)
    plot_graph(graph, os.path.join(args.save_dir, "latent_graph.png"))

    cfgs = _episode_configs(args.n_episodes)
    want = set(args.rungs.split(","))
    env = TwoRoomsEnv(seed=0)
    metrics = {}

    if "0" in want:
        rng = np.random.RandomState(0)
        res = [run_baseline_episode(TwoRoomsEnv(seed=s), random_policy(rng), sr, gr, s) for sr, gr, s in cfgs]
        metrics["rung0_random"] = summarize(res); print("rung0", metrics["rung0_random"])
    if "2" in want:
        res = [run_world_model_episode(model, decode_fn, TwoRoomsEnv(seed=s), device, sr, gr, s,
                                       strategy="greedy", **cem) for sr, gr, s in cfgs]
        metrics["rung2_operative_greedy"] = summarize(res); print("rung2", metrics["rung2_operative_greedy"])
    if "4a" in want:
        res = [run_world_model_episode(model, decode_fn, TwoRoomsEnv(seed=s), device, sr, gr, s,
                                       strategy="waypoint", **cem) for sr, gr, s in cfgs]
        metrics["rung4a_strategic_doorgoal"] = summarize(res); print("rung4a", metrics["rung4a_strategic_doorgoal"])
    if "4b" in want:
        res = [run_graph_episode(model, decode_fn, graph, TwoRoomsEnv(seed=s), device, sr, gr, s, **cem)
               for sr, gr, s in cfgs]
        metrics["rung4b_latent_graph"] = summarize(res); print("rung4b", metrics["rung4b_latent_graph"])
    if "5" in want:
        res = [run_baseline_episode(TwoRoomsEnv(seed=s), heuristic_oracle_policy(), sr, gr, s) for sr, gr, s in cfgs]
        metrics["rung5_oracle"] = summarize(res); print("rung5", metrics["rung5_oracle"])

    report = {"n_episodes": args.n_episodes, "frame_skip": args.frame_skip,
              "decoder_error_world_units": g1["held_out_decode_error_world_units"],
              "graph_k": args.graph_k, "rungs": metrics}
    out = os.path.join(args.save_dir, "ladder_metrics.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    plot_ladder(metrics, os.path.join(args.save_dir, "ladder_success.png"))
    print(f"\n[report] {out}")


if __name__ == "__main__":
    main()
