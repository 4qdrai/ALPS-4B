"""
Generate COMPLEX (4-room, key-gated, NO hazards) training data.

Without hazards the hand-coded heuristic solves ~0% (it oscillates at doors), so
it produces no successful key->door->goal trajectories for the latent graph to
learn from. This generator uses the BFS optimal planner for a fraction of episodes
(clean key-routing demonstrations) mixed with random-momentum exploration (for
node coverage), saving in the standard dataset format.

USAGE
  PYTHONPATH=src python -m alps.benchmarks.two_rooms.generate_complex \
      --save-path data/two_rooms/trajectories_complex.pt --num-episodes 2500 --bfs-fraction 0.5
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, "src")
import argparse, time
import numpy as np
import torch

from alps.benchmarks.two_rooms.environment import TwoRoomsEnv
from alps.benchmarks.two_rooms.optimal_planner import bfs_actions


def generate(save_path, n_episodes=2500, bfs_fraction=0.5, max_steps=120, seed=42):
    rng = np.random.RandomState(seed)
    obs_l, act_l, pos_l, room_l, key_l, starts = [], [], [], [], [], []
    gstep, solved = 0, 0
    t0 = time.time()
    for ep in range(n_episodes):
        s = seed * 100000 + ep
        env = TwoRoomsEnv(seed=s, complex_mode=True, hazards=False)
        obs = env.reset()
        starts.append(gstep)
        plan, pi = None, 0
        if rng.rand() < bfs_fraction:
            planv = TwoRoomsEnv(seed=s, complex_mode=True, hazards=False); planv.reset()
            plan = bfs_actions(planv)
        prev_a = None
        for step in range(max_steps):
            obs_l.append(np.transpose(obs["image"], (2, 0, 1)))
            pos_l.append(obs["position"].copy()); room_l.append(obs["room_id"])
            key_l.append(float(obs.get("has_key", 0.0)))
            if plan is not None and pi < len(plan):
                a = plan[pi]; pi += 1
            else:
                a = prev_a if (prev_a is not None and rng.rand() < 0.6) else rng.randint(0, 4)
                prev_a = a
            act_l.append(int(a))
            obs, _, done, _ = env.step(int(a)); gstep += 1
            if done:
                solved += 1
                break
        if (ep + 1) % 500 == 0:
            print(f"  ep {ep+1}/{n_episodes} | frames {gstep:,} | solved {solved} "
                  f"| {time.time()-t0:.0f}s")
    # Memory-safe packing: preallocate the uint8 output and move frames in (dropping
    # each source reference) instead of np.stack(...).astype(...), which holds the
    # list + stacked copy + astype copy simultaneously (~3x peak RAM -> OOM at scale).
    N = len(obs_l)
    if N:
        obs_buf = np.empty((N, *obs_l[0].shape), dtype=np.uint8)
        for i in range(N):
            obs_buf[i] = obs_l[i]
            obs_l[i] = None
        obs_l.clear()
        observations = torch.from_numpy(obs_buf)
    else:
        observations = torch.empty((0, 3, 128, 128), dtype=torch.uint8)
    res = {
        "observations": observations,
        "actions": torch.tensor(act_l, dtype=torch.long),
        "positions": torch.from_numpy(np.stack(pos_l).astype("float32")) if N else torch.empty((0, 2)),
        "room_ids": torch.tensor(room_l, dtype=torch.long),
        "has_keys": torch.tensor(key_l, dtype=torch.float32),
        "episode_starts": starts,
    }
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(res, save_path)
    print(f"[saved] {save_path} {tuple(res['observations'].shape)} | "
          f"{solved}/{n_episodes} episodes solved ({100*solved/n_episodes:.0f}%)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-path", default="data/two_rooms/trajectories_complex.pt")
    ap.add_argument("--num-episodes", type=int, default=2500)
    ap.add_argument("--bfs-fraction", type=float, default=0.5)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    generate(a.save_path, a.num_episodes, a.bfs_fraction, a.max_steps, a.seed)


if __name__ == "__main__":
    main()
