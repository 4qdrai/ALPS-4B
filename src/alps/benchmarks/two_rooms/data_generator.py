"""
Trajectory data generator for the Two Rooms benchmark.

Generates a dataset of agent trajectories using a mix of random-with-momentum
and heuristic (navigate-to-door-then-target) policies.

Usage:
    python -m alps.benchmarks.two_rooms.data_generator
    # or directly:
    python src/alps/benchmarks/two_rooms/data_generator.py

Output:
    data/two_rooms/trajectories.pt  — dict with keys:
        observations  : uint8  tensor  [N, 3, 128, 128]
        actions       : int64  tensor  [N]
        positions     : float32 tensor [N, 2]
        room_ids      : int64  tensor  [N]
        episode_starts: list[int]  (indices where new episodes begin)
"""

import sys
import os
import time
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
    """Navigate toward the door center [5.0, 5.0], then toward the target.

    Adds 20% random noise (i.e., 20% of the time picks a random action
    instead of the greedy one).
    """

    DOOR_CENTER = np.array([5.0, 5.0], dtype=np.float32)
    NOISE_PROB = 0.2

    def __init__(self, rng: np.random.RandomState):
        self.rng = rng

    def reset(self) -> None:
        pass

    def __call__(self, obs: Dict[str, Any]) -> int:
        # Exploration noise
        if self.rng.rand() < self.NOISE_PROB:
            return self.rng.randint(0, TwoRoomsEnv.NUM_ACTIONS)

        agent_pos = obs["position"]
        target_pos = obs["target"]
        agent_room = TwoRoomsEnv.get_room_id(agent_pos)
        target_room = TwoRoomsEnv.get_room_id(target_pos)

        if agent_room != target_room:
            # Navigate toward the door first
            goal = self.DOOR_CENTER
        else:
            # Same room → go straight to target
            goal = target_pos

        return self._direction_to_action(agent_pos, goal)

    @staticmethod
    def _direction_to_action(pos: np.ndarray, goal: np.ndarray) -> int:
        """Pick the discrete action that moves closest to the goal direction.

        Actions: 0=up(+y), 1=down(-y), 2=left(-x), 3=right(+x)
        """
        diff = goal - pos  # [dx, dy]
        dx, dy = diff[0], diff[1]

        # Pick the axis with the larger absolute difference
        if abs(dy) >= abs(dx):
            return 0 if dy > 0 else 1  # up / down
        else:
            return 3 if dx > 0 else 2  # right / left


# ---------------------------------------------------------------------------
#  Trajectory Generator
# ---------------------------------------------------------------------------

class TrajectoryGenerator:
    """Generate trajectories from TwoRoomsEnv with mixed policies.

    Args:
        num_episodes: total episodes to generate.
        max_steps:    maximum steps per episode.
        heuristic_fraction: fraction of episodes using heuristic policy.
        seed:         RNG seed for reproducibility.
    """

    def __init__(
        self,
        num_episodes: int = 5000,
        max_steps: int = 100,
        heuristic_fraction: float = 0.30,
        seed: int = 42,
    ):
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.heuristic_fraction = heuristic_fraction
        self.seed = seed

    def generate(self) -> Dict[str, Any]:
        """Run all episodes and collect transitions.

        Returns:
            dict with:
                observations  : uint8   [N, 3, 128, 128]
                actions       : int64   [N]
                positions     : float32 [N, 2]
                room_ids      : int64   [N]
                episode_starts: list[int]
        """
        rng = np.random.RandomState(self.seed)
        env = TwoRoomsEnv(seed=self.seed)

        random_policy = RandomMomentumPolicy(rng)
        heuristic_policy = HeuristicPolicy(rng)

        # Pre-allocate lists (will be stacked at the end)
        all_obs: List[np.ndarray] = []
        all_actions: List[int] = []
        all_positions: List[np.ndarray] = []
        all_room_ids: List[int] = []
        episode_starts: List[int] = []

        t_start = time.time()
        global_step = 0

        for ep in range(self.num_episodes):
            # Pick policy for this episode
            use_heuristic = rng.rand() < self.heuristic_fraction
            policy = heuristic_policy if use_heuristic else random_policy
            policy.reset()

            obs = env.reset()
            episode_starts.append(global_step)

            for step_i in range(self.max_steps):
                # Store the current observation *before* acting
                # Convert HWC → CHW for PyTorch convention
                img_chw = np.transpose(obs["image"], (2, 0, 1))  # (3, 128, 128) uint8
                all_obs.append(img_chw)
                all_positions.append(obs["position"].copy())
                all_room_ids.append(obs["room_id"])

                action = policy(obs)
                all_actions.append(action)

                obs, reward, done, info = env.step(action)
                global_step += 1

                if done:
                    break

            # Progress reporting
            if (ep + 1) % 500 == 0:
                elapsed = time.time() - t_start
                print(
                    f"[TrajectoryGenerator] Episode {ep + 1}/{self.num_episodes}  "
                    f"| steps so far: {global_step:,}  "
                    f"| elapsed: {elapsed:.1f}s"
                )

        # --- Pack into tensors ---------------------------------------------------
        print(f"[TrajectoryGenerator] Packing {global_step:,} transitions into tensors …")

        observations = torch.from_numpy(np.stack(all_obs, axis=0))       # uint8 [N, 3, 128, 128]
        actions = torch.tensor(all_actions, dtype=torch.long)            # [N]
        positions = torch.from_numpy(np.stack(all_positions, axis=0))    # float32 [N, 2]
        room_ids = torch.tensor(all_room_ids, dtype=torch.long)          # [N]

        total_time = time.time() - t_start
        print(
            f"[TrajectoryGenerator] Done.  "
            f"N={observations.shape[0]:,}  "
            f"obs={observations.shape}  dtype={observations.dtype}  "
            f"total time: {total_time:.1f}s"
        )

        return {
            "observations": observations,    # uint8 to save memory
            "actions": actions,
            "positions": positions,
            "room_ids": room_ids,
            "episode_starts": episode_starts,
        }


# ---------------------------------------------------------------------------
#  Convenience function
# ---------------------------------------------------------------------------

def generate_dataset(
    save_path: str = "data/two_rooms/trajectories.pt",
    num_episodes: int = 5000,
    max_steps: int = 100,
    heuristic_fraction: float = 0.30,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate the Two Rooms trajectory dataset and save to disk.

    Args:
        save_path:           where to write the .pt file.
        num_episodes:        number of episodes.
        max_steps:           max steps per episode.
        heuristic_fraction:  fraction of episodes using the heuristic policy.
        seed:                RNG seed.

    Returns:
        The dataset dict (same object that was saved).
    """
    generator = TrajectoryGenerator(
        num_episodes=num_episodes,
        max_steps=max_steps,
        heuristic_fraction=heuristic_fraction,
        seed=seed,
    )
    dataset = generator.generate()

    # Ensure output directory exists
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
    generate_dataset()
