"""
Trajectory data generator for the Two Rooms benchmark.

Generates a dataset of agent trajectories using a mix of random-with-momentum
and heuristic policies. Supports both baseline and complex modes.

Usage:
    python -m alps.benchmarks.two_rooms.data_generator
    python -m alps.benchmarks.two_rooms.data_generator --complex-mode
"""

import sys
import os
import time
import argparse
import numpy as np
import torch
from typing import Dict, List, Optional, Any

# Ensure project root is on path when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from alps.benchmarks.two_rooms.environment import TwoRoomsEnv


# ---------------------------------------------------------------------------
#  Policies
# ---------------------------------------------------------------------------

class RandomMomentumPolicy:
    """Random action selection with 60% chance of repeating the last action."""

    MOMENTUM_PROB = 0.6

    def __init__(self, rng: np.random.RandomState):
        self.rng = rng
        self.prev_action: Optional[int] = None

    def reset(self) -> None:
        self.prev_action = None

    def __call__(self, obs: Dict[str, Any]) -> int:
        if self.prev_action is not None and self.rng.rand() < self.MOMENTUM_PROB:
            return self.prev_action
        action = self.rng.randint(0, TwoRoomsEnv.NUM_ACTIONS)
        self.prev_action = action
        return action


class HeuristicPolicy:
    """Navigate toward goals depending on baseline/complex mode configurations.

    Adds 20% random noise (i.e., 20% of the time picks a random action
    instead of the greedy one).
    """

    DOOR_CENTER = np.array([5.0, 5.0], dtype=np.float32)
    NOISE_PROB = 0.2

    def __init__(self, rng: np.random.RandomState, complex_mode: bool = False):
        self.rng = rng
        self.complex_mode = complex_mode

    def reset(self) -> None:
        pass

    def __call__(self, obs: Dict[str, Any]) -> int:
        # Exploration noise
        if self.rng.rand() < self.NOISE_PROB:
            return self.rng.randint(0, TwoRoomsEnv.NUM_ACTIONS)

        agent_pos = obs["position"]
        target_pos = obs["target"]

        if not self.complex_mode:
            # --- Baseline Mode Navigation ---
            agent_room = TwoRoomsEnv.get_room_id(agent_pos, complex_mode=False)
            target_room = TwoRoomsEnv.get_room_id(target_pos, complex_mode=False)

            if agent_room != target_room:
                # Navigate toward the central vertical door first
                goal = self.DOOR_CENTER
            else:
                # Same room → go straight to target
                goal = target_pos
        else:
            # --- Complex Mode Navigation ---
            has_key = bool(obs.get("has_key", 0.0))
            key_pos = obs.get("key_pos", np.array([2.5, 7.5], dtype=np.float32))
            room = TwoRoomsEnv.get_room_id(agent_pos, complex_mode=True)

            if not has_key:
                # Stage 1: Search and retrieve the key
                key_room = TwoRoomsEnv.get_room_id(key_pos, complex_mode=True)
                if room == key_room:
                    goal = key_pos
                elif room == 0:
                    # Squeeze up through horizontal left door to Room 1
                    goal = np.array([2.5, 5.0], dtype=np.float32)
                elif room == 1 and key_room == 2:
                    # Squeeze right through vertical door (open from left) to Room 2
                    goal = np.array([5.0, 5.0], dtype=np.float32)
                else:
                    goal = key_pos
            else:
                # Stage 2: Squeeze down/right to Target in Room 3
                target_room = TwoRoomsEnv.get_room_id(target_pos, complex_mode=True)
                if room == target_room:
                    goal = target_pos
                elif room == 1:
                    # Go down through horizontal left door to Room 0
                    goal = np.array([2.5, 5.0], dtype=np.float32)
                elif room == 2:
                    # Go down through horizontal right door to Room 3
                    goal = np.array([7.5, 5.0], dtype=np.float32)
                elif room == 0:
                    # Navigate through unlocked vertical door to Room 3
                    goal = np.array([5.0, 5.0], dtype=np.float32)
                else:
                    goal = target_pos

        return self._direction_to_action(agent_pos, goal)

    @staticmethod
    def _direction_to_action(pos: np.ndarray, goal: np.ndarray) -> int:
        """Pick the discrete action that moves closest to the goal direction."""
        diff = goal - pos  # [dx, dy]
        dx, dy = diff[0], diff[1]

        if abs(dy) >= abs(dx):
            return 0 if dy > 0 else 1  # up / down
        else:
            return 3 if dx > 0 else 2  # right / left


# ---------------------------------------------------------------------------
#  Trajectory Generator
# ---------------------------------------------------------------------------

class TrajectoryGenerator:
    """Generate trajectories from TwoRoomsEnv with mixed policies."""

    def __init__(
        self,
        num_episodes: int = 5000,
        max_steps: int = 100,
        heuristic_fraction: float = 0.30,
        seed: int = 42,
        complex_mode: bool = False,
        egocentric: bool = False,
        perception_radius: float = None,
    ):
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.heuristic_fraction = heuristic_fraction
        self.seed = seed
        self.complex_mode = complex_mode
        self.egocentric = egocentric
        self.perception_radius = perception_radius

    def _rollout(self, obs_buf: Optional[np.ndarray] = None):
        """One deterministic pass over all episodes (fixed seed → identical
        trajectories every call). Always collects the small per-step arrays. If
        `obs_buf` is given, frames are written straight into it at row `global_step`
        (no intermediate Python list). Returns the small arrays + total frame count.
        """
        rng = np.random.RandomState(self.seed)
        env = TwoRoomsEnv(seed=self.seed, complex_mode=self.complex_mode, egocentric=self.egocentric,
                          perception_radius=self.perception_radius)
        random_policy = RandomMomentumPolicy(rng)
        heuristic_policy = HeuristicPolicy(rng, complex_mode=self.complex_mode)

        all_actions: List[int] = []
        all_positions: List[np.ndarray] = []
        all_room_ids: List[int] = []
        all_has_keys: List[float] = []
        episode_starts: List[int] = []

        t_start = time.time()
        global_step = 0
        for ep in range(self.num_episodes):
            use_heuristic = rng.rand() < self.heuristic_fraction
            policy = heuristic_policy if use_heuristic else random_policy
            policy.reset()

            obs = env.reset()
            episode_starts.append(global_step)

            for step_i in range(self.max_steps):
                if obs_buf is not None:
                    obs_buf[global_step] = np.transpose(obs["image"], (2, 0, 1))  # write in place
                all_positions.append(obs["position"].copy())
                all_room_ids.append(obs["room_id"])
                if self.complex_mode:
                    all_has_keys.append(obs["has_key"])

                action = policy(obs)
                all_actions.append(action)

                obs, reward, done, info = env.step(action)
                global_step += 1
                if done:
                    break

            if (ep + 1) % 500 == 0:
                phase = "fill" if obs_buf is not None else "count"
                print(f"[TrajectoryGenerator] ({phase}) episode {ep + 1}/{self.num_episodes} "
                      f"| steps so far: {global_step:,} | elapsed: {time.time() - t_start:.1f}s")

        return all_actions, all_positions, all_room_ids, all_has_keys, episode_starts, global_step

    def generate(self) -> Dict[str, Any]:
        """Run all episodes and collect transitions.

        TWO-PASS, memory-bounded: the frame buffer is the only large allocation that
        ever exists (~N·48 KB; e.g. 675k frames ≈ 33 GB). Holding a Python list of
        frames AND stacking it would need ~2× that simultaneously — and because each
        48 KB frame is below glibc's mmap threshold, freeing list entries does NOT
        return RSS to the OS, so the naive "free as you go" does not help. Instead:
          pass 1 — counts frames + collects the (tiny) action/position/room arrays;
          pass 2 — fills one preallocated uint8 buffer in place (no list).
        The env is deterministic under the fixed seed, so both passes are identical.
        Cost: 2× the (cheap) rollout; benefit: peak RAM stays at 1× the dataset.
        """
        t0 = time.time()
        print(f"[TrajectoryGenerator] pass 1/2: counting frames over {self.num_episodes} episodes …")
        _, _, _, _, _, N = self._rollout(obs_buf=None)

        gb = N * 3 * 128 * 128 / 1e9
        print(f"[TrajectoryGenerator] pass 2/2: allocating {N:,} × 3×128×128 uint8 (~{gb:.1f} GB) and filling …")
        obs_buf = np.empty((N, 3, 128, 128), dtype=np.uint8) if N else np.empty((0, 3, 128, 128), np.uint8)
        all_actions, all_positions, all_room_ids, all_has_keys, episode_starts, N2 = self._rollout(obs_buf=obs_buf)
        assert N2 == N, f"non-deterministic rollout: {N2} != {N}"

        res = {
            "observations": torch.from_numpy(obs_buf),                       # uint8 [N,3,128,128]
            "actions": torch.tensor(all_actions, dtype=torch.long),
            "positions": torch.from_numpy(np.stack(all_positions, axis=0)) if N else torch.empty((0, 2)),
            "room_ids": torch.tensor(all_room_ids, dtype=torch.long),
            "episode_starts": episode_starts,
        }
        if self.complex_mode:
            res["has_keys"] = torch.tensor(all_has_keys, dtype=torch.float32)

        print(f"[TrajectoryGenerator] Done.  N={N:,}  obs={tuple(res['observations'].shape)}  "
              f"total time: {time.time() - t0:.1f}s")
        return res


# ---------------------------------------------------------------------------
#  Convenience function
# ---------------------------------------------------------------------------

def generate_dataset(
    save_path: str = "data/two_rooms/trajectories.pt",
    num_episodes: int = 5000,
    max_steps: int = 100,
    heuristic_fraction: float = 0.30,
    seed: int = 42,
    complex_mode: bool = False,
    egocentric: bool = False,
    perception_radius: float = None,
) -> Dict[str, Any]:
    """Generate the dataset and save to disk."""
    generator = TrajectoryGenerator(
        num_episodes=num_episodes,
        max_steps=max_steps,
        heuristic_fraction=heuristic_fraction,
        seed=seed,
        complex_mode=complex_mode,
        egocentric=egocentric,
        perception_radius=perception_radius,
    )
    dataset = generator.generate()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    print(f"[generate_dataset] Saving to {save_path} …")
    torch.save(dataset, save_path)
    file_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"[generate_dataset] Saved ({file_mb:.1f} MB)")

    return dataset


# ---------------------------------------------------------------------------
#  CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two Rooms Dataset Generator")
    parser.add_argument("--save-path", type=str, default="data/two_rooms/trajectories.pt")
    parser.add_argument("--num-episodes", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--heuristic-fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--complex-mode", action="store_true", help="Enable 4-room complex navigation mode")
    parser.add_argument("--egocentric", action="store_true",
                        help="Agent-centered rendering (world scrolls around the agent) so the "
                             "pure-SSL predictor is forced to learn controllable dynamics.")
    parser.add_argument("--perception-radius", type=float, default=None,
                        help="Limited perception (egocentric): the agent observes only a disk of "
                             "this radius (world units) around itself; beyond is unobserved.")
    args = parser.parse_args()

    generate_dataset(
        save_path=args.save_path,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        heuristic_fraction=args.heuristic_fraction,
        seed=args.seed,
        complex_mode=args.complex_mode,
        egocentric=args.egocentric,
        perception_radius=args.perception_radius,
    )
