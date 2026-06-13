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


def _rollout(n_episodes, bfs_fraction, max_steps, seed, obs_buf=None):
    """One deterministic pass over all episodes (fixed seed + deterministic BFS →
    identical trajectories every call). Collects the small per-step arrays; if
    `obs_buf` is given, writes each frame straight into it (no Python frame list).
    Returns the small arrays, total frame count, and #solved."""
    rng = np.random.RandomState(seed)
    act_l, pos_l, room_l, key_l, starts = [], [], [], [], []
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
            if obs_buf is not None:
                obs_buf[gstep] = np.transpose(obs["image"], (2, 0, 1))   # write in place
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
            phase = "fill" if obs_buf is not None else "count"
            print(f"  ({phase}) ep {ep+1}/{n_episodes} | frames {gstep:,} | solved {solved} "
                  f"| {time.time()-t0:.0f}s")
    return act_l, pos_l, room_l, key_l, starts, gstep, solved


def generate(save_path, n_episodes=2500, bfs_fraction=0.5, max_steps=120, seed=42):
    # TWO-PASS, memory-bounded (see data_generator.py): the frame buffer is the only
    # large allocation. A Python frame list + np.stack(...).astype(...) would hold
    # ~2-3× the dataset at once, and 48 KB frames are below glibc's mmap threshold so
    # freeing list entries does NOT return RSS — that OOM-killed 10k-ep generation.
    print(f"[generate_complex] pass 1/2: counting frames over {n_episodes} episodes …")
    _, _, _, _, _, N, _ = _rollout(n_episodes, bfs_fraction, max_steps, seed, obs_buf=None)
    gb = N * 3 * 128 * 128 / 1e9
    print(f"[generate_complex] pass 2/2: allocating {N:,} × 3×128×128 uint8 (~{gb:.1f} GB) and filling …")
    obs_buf = np.empty((N, 3, 128, 128), dtype=np.uint8) if N else np.empty((0, 3, 128, 128), np.uint8)
    act_l, pos_l, room_l, key_l, starts, N2, solved = _rollout(
        n_episodes, bfs_fraction, max_steps, seed, obs_buf=obs_buf)
    assert N2 == N, f"non-deterministic rollout: {N2} != {N}"

    res = {
        "observations": torch.from_numpy(obs_buf),
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
