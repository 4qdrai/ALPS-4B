"""
ALPS-4B Two Rooms — world-model planning, oracle/random baselines, and eval harness.

This module turns a gate-passing world model (encoder + action-conditioned
operative predictor + position decoder, see `alps.evaluation.repr_decoder_gate`)
into an actual controller, and provides the baselines needed to put its numbers
in context (the ablation ladder lives in `run_ablation_ladder.py`).

Planners
--------
WorldModelMPC
    Position-space Cross-Entropy-Method MPC. At each env step it:
      1. encodes the current frame -> z_t,
      2. samples candidate discrete action sequences,
      3. rolls them forward in latent space with the operative predictor,
      4. DECODES every rolled latent to (x, y) with the validated probe,
      5. scores by negative distance of the decoded endpoint to the active
         sub-goal, minus a wall-crossing penalty,
      6. executes the first action and replans (MPC).
    Two sub-goal strategies:
      * "greedy"   : sub-goal is always the final goal (operative-only; this is
                     expected to get stuck at the wall on cross-room episodes —
                     the local minimum that motivates the hierarchy).
      * "waypoint" : a strategic plan supplies an ordered list of sub-goals
                     (e.g. [door, goal] or a latent-graph shortest path); the
                     planner advances to the next sub-goal once the current one
                     is reached. This is where the strategic layer adds value.

Baselines
---------
heuristic_oracle_policy : near-optimal navigator on TRUE state (rung 5 ceiling,
                          also used to estimate optimal path length for SPL).
random_policy           : random-with-momentum (rung 0 floor).

Metrics
-------
Success rate (same/cross/overall), SPL (success weighted by path-length
optimality), mean path length, mean System-2 / replan count.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from alps.benchmarks.two_rooms.environment import TwoRoomsEnv


# ---------------------------------------------------------------------------
#  Geometry helpers (baseline two-rooms; complex handled by latent graph)
# ---------------------------------------------------------------------------

DOOR_XY = np.array([5.0, 5.0], dtype=np.float32)
WALL_X = 5.0
DOOR_Y_MIN, DOOR_Y_MAX = 4.5, 5.5


def crosses_wall(p0: np.ndarray, p1: np.ndarray) -> bool:
    """True if the segment p0->p1 crosses the vertical wall outside the door gap."""
    x0, x1 = float(p0[0]), float(p1[0])
    if (x0 < WALL_X) == (x1 < WALL_X):
        return False  # same side
    # crossing point y at x = WALL_X
    if abs(x1 - x0) < 1e-8:
        return False
    t = (WALL_X - x0) / (x1 - x0)
    y_cross = float(p0[1]) + t * (float(p1[1]) - float(p0[1]))
    return not (DOOR_Y_MIN <= y_cross <= DOOR_Y_MAX)


def obs_to_frame(obs: dict, device) -> torch.Tensor:
    img = torch.from_numpy(obs["image"]).permute(2, 0, 1).float() / 255.0
    return img.to(device)


# ---------------------------------------------------------------------------
#  World-model MPC planner (position space)
# ---------------------------------------------------------------------------

class WorldModelMPC:
    NUM_ACTIONS = 4

    @staticmethod
    @torch.no_grad()
    def _cem_actions(
        model, decode_fn, frame: torch.Tensor, subgoal_xy: np.ndarray,
        horizon: int, n_cand: int, n_elite: int, n_iter: int,
        wall_penalty: float, device,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Return the best action sequence [horizon] and its decoded endpoint xy."""
        z0 = model.encode_frame(frame.unsqueeze(0))  # [1,N,D]
        goal = torch.tensor(subgoal_xy, device=device, dtype=torch.float32)

        probs = torch.ones(horizon, WorldModelMPC.NUM_ACTIONS, device=device)
        probs /= WorldModelMPC.NUM_ACTIONS
        best_actions, best_score, best_xy = None, -1e30, None

        for _ in range(n_iter):
            cand = torch.zeros(n_cand, horizon, dtype=torch.long, device=device)
            for t in range(horizon):
                cand[:, t] = torch.distributions.Categorical(probs=probs[t]).sample((n_cand,))
            oh = F.one_hot(cand, WorldModelMPC.NUM_ACTIONS).float()  # [C,H,4]

            z = z0.expand(n_cand, -1, -1)  # [C,N,D]
            decoded = []  # per-step decoded xy
            for t in range(horizon):
                z = model.predict_next(z, oh[:, t])
                decoded.append(decode_fn(z))  # [C,2]
            decoded = torch.stack(decoded, dim=1)  # [C,H,2]

            endpoint = decoded[:, -1]                       # [C,2]
            dist = (endpoint - goal).norm(dim=1)            # [C]
            # wall penalty: count decoded segments crossing the wall outside the door
            xs = decoded[:, :, 0]
            ys = decoded[:, :, 1]
            x_prev = torch.cat([torch.full((n_cand, 1), float(decode_fn(z0)[0, 0]),
                                           device=device), xs[:, :-1]], dim=1)
            y_prev = torch.cat([torch.full((n_cand, 1), float(decode_fn(z0)[0, 1]),
                                           device=device), ys[:, :-1]], dim=1)
            side_change = ((x_prev < WALL_X) != (xs < WALL_X)).float()
            # crossing y interpolation
            denom = (xs - x_prev)
            denom = torch.where(denom.abs() < 1e-6, torch.full_like(denom, 1e-6), denom)
            tcross = (WALL_X - x_prev) / denom
            ycross = y_prev + tcross * (ys - y_prev)
            bad_cross = side_change * ((ycross < DOOR_Y_MIN) | (ycross > DOOR_Y_MAX)).float()
            penalty = wall_penalty * bad_cross.sum(dim=1)   # [C]

            scores = -(dist + penalty)
            elite_idx = torch.topk(scores, min(n_elite, n_cand)).indices
            top = scores.argmax().item()
            if scores[top].item() > best_score:
                best_score = scores[top].item()
                best_actions = cand[top].clone()
                best_xy = endpoint[top].cpu().numpy()

            # refit categorical from elite frequencies (Laplace-smoothed)
            new_probs = torch.zeros_like(probs)
            elite = cand[elite_idx]
            for t in range(horizon):
                for a in range(WorldModelMPC.NUM_ACTIONS):
                    new_probs[t, a] = (elite[:, t] == a).float().sum()
                new_probs[t] = (new_probs[t] + 1.0) / (elite.shape[0] + WorldModelMPC.NUM_ACTIONS)
            probs = new_probs

        return best_actions, best_xy


# ---------------------------------------------------------------------------
#  Baseline policies
# ---------------------------------------------------------------------------

def random_policy(rng: np.random.RandomState):
    state = {"prev": None}

    def policy(obs):
        if state["prev"] is not None and rng.rand() < 0.6:
            return state["prev"]
        a = rng.randint(0, 4)
        state["prev"] = a
        return a
    return policy


def _greedy_action_to(pos: np.ndarray, goal: np.ndarray) -> int:
    diff = goal - pos
    if abs(diff[1]) >= abs(diff[0]):
        return 0 if diff[1] > 0 else 1
    return 3 if diff[0] > 0 else 2


def heuristic_oracle_policy():
    """Near-optimal: route via the door when cross-room, else straight to goal."""
    def policy(obs):
        pos, goal = obs["position"], obs["target"]
        if TwoRoomsEnv.get_room_id(pos) != TwoRoomsEnv.get_room_id(goal):
            sub = DOOR_XY if abs(pos[0] - WALL_X) > 0.4 or not (DOOR_Y_MIN <= pos[1] <= DOOR_Y_MAX) else goal
        else:
            sub = goal
        return _greedy_action_to(pos, sub)
    return policy


# ---------------------------------------------------------------------------
#  Strategic sub-goal plan (baseline two-rooms): [door, goal] if cross-room
# ---------------------------------------------------------------------------

def strategic_waypoints(start_xy: np.ndarray, goal_xy: np.ndarray) -> List[np.ndarray]:
    if TwoRoomsEnv.get_room_id(start_xy) != TwoRoomsEnv.get_room_id(goal_xy):
        return [DOOR_XY.copy(), goal_xy.copy()]
    return [goal_xy.copy()]


# ---------------------------------------------------------------------------
#  Episode rollout + evaluation
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    success: bool
    path_len: int
    optimal_len: int
    replans: int
    is_cross: bool


@torch.no_grad()
def greedy_step_action(model, decode_fn, frame: torch.Tensor, subgoal_xy: np.ndarray, device) -> int:
    """One-step decoded-greedy: pick the action whose DECODED next position is
    closest to the sub-goal. Uses only the validated 1-step dynamics (G2) and
    re-encodes the true frame every step, avoiding multi-step latent drift."""
    z = model.encode_frame(frame.unsqueeze(0))           # [1,N,D]
    goal = torch.tensor(subgoal_xy, device=device, dtype=torch.float32)
    best_a, best_d = 0, float("inf")
    for a in range(WorldModelMPC.NUM_ACTIONS):
        oh = F.one_hot(torch.tensor([a], device=device), WorldModelMPC.NUM_ACTIONS).float()
        pos = decode_fn(model.predict_next(z, oh))[0]    # [2]
        d = float((pos - goal).norm().item())
        if d < best_d:
            best_d, best_a = d, a
    return best_a


def run_world_model_episode(
    model, decode_fn, env: TwoRoomsEnv, device,
    start_room: int, goal_room: int, seed: int,
    strategy: str = "waypoint",
    horizon: int = 15, n_cand: int = 100, n_elite: int = 10, n_iter: int = 8,
    replan_interval: int = 4, max_steps: int = 150, wall_penalty: float = 3.0,
    reach_radius: float = 0.6,
) -> EpisodeResult:
    obs = env.reset(start_room=start_room, goal_room=goal_room)
    start_xy = obs["position"].copy()
    goal_xy = obs["target"].copy()
    is_cross = start_room != goal_room

    # optimal length from the oracle (for SPL)
    optimal_len = _oracle_path_len(start_room, goal_room, seed)

    if strategy == "waypoint":
        waypoints = strategic_waypoints(start_xy, goal_xy)
    else:
        waypoints = [goal_xy.copy()]
    wp_idx, steps, reached = 0, 0, False

    while steps < max_steps and not reached:
        sub = waypoints[min(wp_idx, len(waypoints) - 1)]
        a = greedy_step_action(model, decode_fn, obs_to_frame(obs, device), sub, device)
        obs, _, done, info = env.step(a)
        steps += 1
        cur = obs["position"]
        if wp_idx < len(waypoints) - 1 and np.linalg.norm(cur - waypoints[wp_idx]) < reach_radius:
            wp_idx += 1
        if done or info["distance"] < reach_radius:
            reached = True
    return EpisodeResult(reached, steps, optimal_len, replans=steps, is_cross=is_cross)


def _oracle_path_len(start_room: int, goal_room: int, seed: int, max_steps: int = 150) -> int:
    env = TwoRoomsEnv(seed=seed)
    obs = env.reset(start_room=start_room, goal_room=goal_room)
    pol = heuristic_oracle_policy()
    for s in range(max_steps):
        a = pol(obs)
        obs, _, done, info = env.step(a)
        if done or info["distance"] < 0.6:
            return s + 1
    return max_steps


def run_baseline_episode(env: TwoRoomsEnv, policy, start_room, goal_room, seed,
                         max_steps=150, reach_radius=0.6) -> EpisodeResult:
    obs = env.reset(start_room=start_room, goal_room=goal_room)
    optimal_len = _oracle_path_len(start_room, goal_room, seed)
    is_cross = start_room != goal_room
    for s in range(max_steps):
        a = policy(obs)
        obs, _, done, info = env.step(a)
        if done or info["distance"] < reach_radius:
            return EpisodeResult(True, s + 1, optimal_len, 0, is_cross)
    return EpisodeResult(False, max_steps, optimal_len, 0, is_cross)


def summarize(results: List[EpisodeResult]) -> Dict:
    def spl(rs):
        if not rs:
            return 0.0
        return float(np.mean([
            (1.0 if r.success else 0.0) * r.optimal_len / max(r.path_len, r.optimal_len, 1)
            for r in rs
        ]))
    same = [r for r in results if not r.is_cross]
    cross = [r for r in results if r.is_cross]
    return {
        "n": len(results),
        "overall_success": float(np.mean([r.success for r in results])) if results else 0.0,
        "same_room_success": float(np.mean([r.success for r in same])) if same else 0.0,
        "cross_room_success": float(np.mean([r.success for r in cross])) if cross else 0.0,
        "overall_spl": spl(results),
        "cross_room_spl": spl(cross),
        "avg_path_len": float(np.mean([r.path_len for r in results])) if results else 0.0,
        "avg_replans": float(np.mean([r.replans for r in results])) if results else 0.0,
    }
