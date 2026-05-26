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
    ):
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.heuristic_fraction = heuristic_fraction
        self.seed = seed
        self.complex_mode = complex_mode

    def generate(self) -> Dict[str, Any]:
        """Run all episodes and collect transitions."""
        rng = np.random.RandomState(self.seed)
        env = TwoRoomsEnv(seed=self.seed, complex_mode=self.complex_mode)

        random_policy = RandomMomentumPolicy(rng)
        heuristic_policy = HeuristicPolicy(rng, complex_mode=self.complex_mode)

        # Pre-allocate lists
        all_obs: List[np.ndarray] = []
        all_actions: List[int] = []
        all_positions: List[np.ndarray] = []
        all_room_ids: List[int] = []
        episode_starts: List[int] = []

        # Complex state lists
        all_has_keys: List[float] = []

        t_start = time.time()
        global_step = 0

        for ep in range(self.num_episodes):
            use_heuristic = rng.rand() < self.heuristic_fraction
            policy = heuristic_policy if use_heuristic else random_policy
            policy.reset()

            obs = env.reset()
            episode_starts.append(global_step)

            for step_i in range(self.max_steps):
                img_chw = np.transpose(obs["image"], (2, 0, 1))  # (3, 128, 128) uint8
                all_obs.append(img_chw)
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
                elapsed = time.time() - t_start
                print(
                    f"[TrajectoryGenerator] Episode {ep + 1}/{self.num_episodes}  "
                    f"| steps so far: {global_step:,}  "
                    f"| elapsed: {elapsed:.1f}s"
                )

        print(f"[TrajectoryGenerator] Packing {global_step:,} transitions into tensors …")

        observations = torch.from_numpy(np.stack(all_obs, axis=0))       # uint8 [N, 3, 128, 128]
        actions = torch.tensor(all_actions, dtype=torch.long)            # [N]
        positions = torch.from_numpy(np.stack(all_positions, axis=0))    # float32 [N, 2]
        room_ids = torch.tensor(all_room_ids, dtype=torch.long)          # [N]

        res = {
            "observations": observations,
            "actions": actions,
            "positions": positions,
            "room_ids": room_ids,
            "episode_starts": episode_starts,
        }

        if self.complex_mode:
            res["has_keys"] = torch.tensor(all_has_keys, dtype=torch.float32)

        total_time = time.time() - t_start
        print(
            f"[TrajectoryGenerator] Done.  "
            f"N={observations.shape[0]:,}  "
            f"obs={observations.shape}  "
            f"total time: {total_time:.1f}s"
        )

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
) -> Dict[str, Any]:
    """Generate the dataset and save to disk."""
    generator = TrajectoryGenerator(
        num_episodes=num_episodes,
        max_steps=max_steps,
        heuristic_fraction=heuristic_fraction,
        seed=seed,
        complex_mode=complex_mode,
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
    args = parser.parse_args()

    generate_dataset(
        save_path=args.save_path,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        heuristic_fraction=args.heuristic_fraction,
        seed=args.seed,
        complex_mode=args.complex_mode,
    )
