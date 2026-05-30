"""
ALPS-4B Two Rooms Benchmark — CEM and Hierarchical Planners.

Implements Cross-Entropy Method (CEM) planning over a trained ALPS model's
latent-space predictor, plus a hierarchical variant that exploits the
strategic VQ codebook for room-aware two-phase navigation.

Key design decisions:
  - Actions are discrete 4-way (up=0, down=1, left=2, right=3), represented
    as one-hot 4D vectors inside the predictor.
  - The CEM maintains a *categorical distribution* (probabilities per time
    step) instead of a Gaussian, matching the discrete action space.
  - Rollouts happen entirely in latent space via
    `operative_layer.predict_next_state`, making planning ~1000x faster
    than pixel-space simulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PlanResult:
    """Container for planner outputs and diagnostics."""
    best_actions: torch.Tensor          # [horizon] int action indices
    best_action_onehots: torch.Tensor   # [horizon, 4] one-hot
    best_score: float                   # negative L2² of best candidate
    elite_scores: List[float] = field(default_factory=list)
    per_iteration_best: List[float] = field(default_factory=list)
    latent_trajectory: Optional[torch.Tensor] = None  # [horizon+1, N, D]
    action_probs_history: Optional[List[torch.Tensor]] = None  # per-iter distributions


@dataclass
class HierarchicalPlanResult:
    """Container for hierarchical planner outputs."""
    phase1_result: Optional[PlanResult]    # navigate to door (if cross-room)
    phase2_result: PlanResult              # navigate to goal
    combined_actions: torch.Tensor         # [total_steps] int action indices
    combined_onehots: torch.Tensor         # [total_steps, 4] one-hot
    is_cross_room: bool
    start_concept_code: int
    goal_concept_code: int
    total_score: float


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def one_hot_actions(action_indices: torch.Tensor, num_actions: int = 4) -> torch.Tensor:
    """Convert integer action indices → one-hot vectors.

    Args:
        action_indices: Long tensor of arbitrary shape, values in [0, num_actions).
        num_actions: Vocabulary size (4 for Two Rooms).

    Returns:
        Float tensor of shape (*action_indices.shape, num_actions).
    """
    return F.one_hot(action_indices.long(), num_classes=num_actions).float()


def encode_observation(model: nn.Module, obs: torch.Tensor) -> torch.Tensor:
    """Encode a single observation frame through the vision encoder.

    The ALPS encoder expects video tensors of shape [B, C, T, H, W].
    If the model has encode_single_frame() (v2 step-wise training), we use it
    for perfect train/eval alignment. Otherwise falls back to broadcast approach.

    Args:
        model: An ALPSModel instance (or any module with an `encoder` attr).
        obs: Single observation, shape [C, H, W] or [B, C, H, W].

    Returns:
        Latent tokens, shape [1, N, D] (batch dim always 1).
    """
    if obs.dim() == 3:
        obs = obs.unsqueeze(0)  # [1, C, H, W]
    
    # v2: Use encode_single_frame for train/eval alignment
    if hasattr(model, 'encode_single_frame'):
        with torch.no_grad():
            z = model.encode_single_frame(obs)  # [B, N, D]
        return z
    
    # Legacy fallback: broadcast to temporal dimension
    B, C, H, W = obs.shape
    T = 8  # Match SINGLE_FRAME_T constant
    video = obs.unsqueeze(2).expand(B, C, T, H, W)   # [B, C, T, H, W]
    with torch.no_grad():
        z = model.encoder(video)  # [B, N, D]
    return z


def rollout_latent(
    model: nn.Module,
    z_start: torch.Tensor,
    action_onehots: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Roll out a sequence of actions through the operative predictor.

    Args:
        model: ALPSModel with `operative_layer.predict_next_state`.
        z_start: Starting latent, shape [1, N, D].
        action_onehots: One-hot action sequence, shape [horizon, 4].

    Returns:
        z_final: Final latent state, shape [1, N, D].
        z_trajectory: Full trajectory, shape [horizon+1, N, D].
    """
    horizon = action_onehots.shape[0]
    z = z_start  # [1, N, D]
    trajectory = [z.squeeze(0)]  # collect [N, D] slices

    for t in range(horizon):
        a = action_onehots[t].unsqueeze(0)  # [1, 4]
        z = model.operative_layer.predict_next_state(z, a)
        trajectory.append(z.squeeze(0))

    z_trajectory = torch.stack(trajectory, dim=0)  # [horizon+1, N, D]
    return z, z_trajectory


# ---------------------------------------------------------------------------
# CEM Planner
# ---------------------------------------------------------------------------

class CEMPlanner:
    """Cross-Entropy Method planner operating in ALPS latent space.

    At each iteration the planner:
      1. Samples candidate action sequences from a categorical distribution.
      2. Rolls each candidate through the operative predictor.
      3. Scores candidates by negative L2² distance to the goal latent.
      4. Updates the distribution toward the elite (top-scoring) candidates.

    The result is an action sequence whose predicted latent endpoint is as
    close as possible to the encoded goal observation.
    """

    NUM_ACTIONS: int = 4  # up, down, left, right

    @staticmethod
    @torch.no_grad()
    def plan(
        model: nn.Module,
        start_obs: torch.Tensor,
        goal_obs: torch.Tensor,
        horizon: int = 30,
        num_candidates: int = 200,
        num_elites: int = 20,
        num_iterations: int = 5,
    ) -> PlanResult:
        """Plan an action sequence to reach *goal_obs* from *start_obs*.

        Args:
            model: Trained ALPSModel.
            start_obs: Start observation [C, H, W] or [1, C, H, W].
            goal_obs: Goal observation [C, H, W] or [1, C, H, W].
            horizon: Planning horizon (number of actions).
            num_candidates: Population size per CEM iteration.
            num_elites: Number of top candidates used for distribution update.
            num_iterations: CEM optimisation iterations.

        Returns:
            PlanResult with the best action sequence and diagnostics.
        """
        device = next(model.parameters()).device

        # --- Step 1: encode start and goal ---
        z_start = encode_observation(model, start_obs.to(device))  # [1, N, D]
        z_goal = encode_observation(model, goal_obs.to(device))    # [1, N, D]

        # --- Step 2: initialise uniform categorical distribution ---
        # action_probs[t, a] = probability of action a at time step t
        action_probs = torch.ones(horizon, CEMPlanner.NUM_ACTIONS, device=device)
        action_probs = action_probs / CEMPlanner.NUM_ACTIONS  # uniform

        per_iteration_best: List[float] = []
        action_probs_history: List[torch.Tensor] = [action_probs.clone()]

        best_global_score = -float("inf")
        best_global_actions: Optional[torch.Tensor] = None
        best_global_trajectory: Optional[torch.Tensor] = None

        for iteration in range(num_iterations):
            # --- Step 3a: sample candidate action sequences ---
            # Categorical sampling from the per-step distribution
            # action_probs: [horizon, 4] → sample [num_candidates, horizon]
            candidates = torch.zeros(num_candidates, horizon, dtype=torch.long, device=device)
            for t in range(horizon):
                dist = torch.distributions.Categorical(probs=action_probs[t])
                candidates[:, t] = dist.sample((num_candidates,))

            # --- Step 3b: roll out each candidate ---
            scores = torch.zeros(num_candidates, device=device)
            trajectories = []

            for c in range(num_candidates):
                act_onehots = one_hot_actions(candidates[c], CEMPlanner.NUM_ACTIONS)  # [H, 4]
                z_final, z_traj = rollout_latent(model, z_start, act_onehots)
                # Score = negative squared L2 distance to goal latent
                score = -(z_final - z_goal).pow(2).sum()
                scores[c] = score
                trajectories.append(z_traj)

            # --- Step 3c: select elites ---
            elite_indices = torch.topk(scores, num_elites).indices
            elite_actions = candidates[elite_indices]  # [E, horizon]
            elite_scores_list = scores[elite_indices].tolist()

            # Track best across all iterations
            iter_best_idx = scores.argmax().item()
            iter_best_score = scores[iter_best_idx].item()
            per_iteration_best.append(iter_best_score)

            if iter_best_score > best_global_score:
                best_global_score = iter_best_score
                best_global_actions = candidates[iter_best_idx].clone()
                best_global_trajectory = trajectories[iter_best_idx].clone()

            # --- Step 3d: update distribution from elite frequencies ---
            new_probs = torch.zeros(horizon, CEMPlanner.NUM_ACTIONS, device=device)
            for t in range(horizon):
                for a in range(CEMPlanner.NUM_ACTIONS):
                    new_probs[t, a] = (elite_actions[:, t] == a).float().sum()
                # Add Laplace smoothing to prevent zero probabilities
                new_probs[t] = (new_probs[t] + 1.0) / (num_elites + CEMPlanner.NUM_ACTIONS)

            action_probs = new_probs
            action_probs_history.append(action_probs.clone())

        # Build return value
        best_onehots = one_hot_actions(best_global_actions, CEMPlanner.NUM_ACTIONS)
        return PlanResult(
            best_actions=best_global_actions,
            best_action_onehots=best_onehots,
            best_score=best_global_score,
            elite_scores=elite_scores_list,
            per_iteration_best=per_iteration_best,
            latent_trajectory=best_global_trajectory,
            action_probs_history=action_probs_history,
        )


# ---------------------------------------------------------------------------
# Hierarchical Planner (strategic VQ-aware)
# ---------------------------------------------------------------------------

class HierarchicalPlanner:
    """Two-phase planner that exploits the VQ codebook for room detection.

    Strategy:
      1. Encode start and goal through the strategic layer's VQ bottleneck
         to obtain their discrete concept codes.
      2. If both positions share the same concept code (same room), run a
         single CEM phase directly toward the goal.
      3. If the codes differ (cross-room), run CEM in two phases:
           Phase 1 — navigate from start to the door region.
           Phase 2 — navigate from the door to the final goal.

    The door position is defined as the center of the 1-unit gap in the wall
    at (x=5, y=5) in the Two Rooms environment.
    """

    # Door center in the 10×10 continuous coordinate system
    DOOR_X: float = 5.0
    DOOR_Y: float = 5.0

    @staticmethod
    def _get_concept_code(model: nn.Module, obs: torch.Tensor) -> int:
        """Encode an observation → strategic VQ code (majority vote across patches).

        Args:
            model: ALPSModel.
            obs: Observation tensor [C, H, W].

        Returns:
            Integer concept code (most frequent VQ index across spatial patches).
        """
        device = next(model.parameters()).device
        z = encode_observation(model, obs.to(device))  # [1, N, D]

        # Pass through operative and strategic layers
        flat_subgoal = torch.zeros_like(z)
        z_op, _ = model.operative_layer(z, flat_subgoal)
        _, _, indices = model.strategic_layer.vq(z_op)  # indices: [1, N]

        # Majority-vote code across spatial patches
        codes = indices[0].cpu()  # [N]
        code = int(torch.mode(codes).values.item())
        return code

    @staticmethod
    def _render_door_obs(resolution: int = 128) -> torch.Tensor:
        """Produce a synthetic observation of the door zone.

        Renders a simple Two Rooms frame with the agent positioned at the
        door center (5.0, 5.0).  Colors follow the project spec:
          - Rooms: light gray (0.75)
          - Walls: dark brown (0.35, 0.22, 0.10)
          - Agent: red (1, 0, 0)
          - Background: dark gray (0.25)

        Returns:
            Observation tensor [3, resolution, resolution].
        """
        img = torch.full((3, resolution, resolution), 0.25)  # dark gray bg

        # Map continuous 10×10 space → pixel coords
        def to_px(x: float, y: float) -> Tuple[int, int]:
            px = int(x / 10.0 * resolution)
            py = int(y / 10.0 * resolution)
            return min(max(px, 0), resolution - 1), min(max(py, 0), resolution - 1)

        # Fill rooms (light gray)
        room_val = 0.75
        for c in range(3):
            img[c, :, :] = room_val

        # Draw wall at x=5 (dark brown)
        wall_px = int(5.0 / 10.0 * resolution)
        wall_thickness = max(2, resolution // 64)
        door_y_lo = int(4.5 / 10.0 * resolution)
        door_y_hi = int(5.5 / 10.0 * resolution)
        brown = torch.tensor([0.35, 0.22, 0.10])
        for dx in range(-wall_thickness, wall_thickness + 1):
            wx = min(max(wall_px + dx, 0), resolution - 1)
            for py in range(resolution):
                if py < door_y_lo or py > door_y_hi:
                    for c in range(3):
                        img[c, py, wx] = brown[c]

        # Draw agent dot at door center (red)
        ax, ay = to_px(HierarchicalPlanner.DOOR_X, HierarchicalPlanner.DOOR_Y)
        radius = max(3, resolution // 32)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    px_x = min(max(ax + dx, 0), resolution - 1)
                    px_y = min(max(ay + dy, 0), resolution - 1)
                    img[0, px_y, px_x] = 1.0
                    img[1, px_y, px_x] = 0.0
                    img[2, px_y, px_x] = 0.0

        return img

    @staticmethod
    @torch.no_grad()
    def plan(
        model: nn.Module,
        start_obs: torch.Tensor,
        goal_obs: torch.Tensor,
        horizon: int = 30,
        num_candidates: int = 200,
        num_elites: int = 20,
        num_iterations: int = 5,
        door_horizon: int = 20,
    ) -> HierarchicalPlanResult:
        """Hierarchical plan: detect room membership then CEM in 1 or 2 phases.

        Args:
            model: Trained ALPSModel.
            start_obs: Start observation [C, H, W].
            goal_obs: Goal observation [C, H, W].
            horizon: CEM horizon for each phase.
            num_candidates: CEM population.
            num_elites: CEM elite count.
            num_iterations: CEM iterations per phase.
            door_horizon: Horizon for Phase 1 (start → door), can be shorter
                because the door is typically closer than the goal.

        Returns:
            HierarchicalPlanResult with combined action sequence.
        """
        start_code = HierarchicalPlanner._get_concept_code(model, start_obs)
        goal_code = HierarchicalPlanner._get_concept_code(model, goal_obs)

        is_cross_room = (start_code != goal_code)

        if not is_cross_room:
            # Same room: single-phase CEM directly to goal
            result = CEMPlanner.plan(
                model, start_obs, goal_obs,
                horizon=horizon,
                num_candidates=num_candidates,
                num_elites=num_elites,
                num_iterations=num_iterations,
            )
            return HierarchicalPlanResult(
                phase1_result=None,
                phase2_result=result,
                combined_actions=result.best_actions,
                combined_onehots=result.best_action_onehots,
                is_cross_room=False,
                start_concept_code=start_code,
                goal_concept_code=goal_code,
                total_score=result.best_score,
            )

        # Cross-room: two-phase CEM
        device = next(model.parameters()).device
        door_obs = HierarchicalPlanner._render_door_obs(
            resolution=start_obs.shape[-1]
        ).to(device)

        # Phase 1: start → door
        phase1 = CEMPlanner.plan(
            model, start_obs, door_obs,
            horizon=door_horizon,
            num_candidates=num_candidates,
            num_elites=num_elites,
            num_iterations=num_iterations,
        )

        # Phase 2: door → goal
        phase2 = CEMPlanner.plan(
            model, door_obs, goal_obs,
            horizon=horizon,
            num_candidates=num_candidates,
            num_elites=num_elites,
            num_iterations=num_iterations,
        )

        # Concatenate action sequences
        combined_actions = torch.cat([phase1.best_actions, phase2.best_actions], dim=0)
        combined_onehots = torch.cat([phase1.best_action_onehots, phase2.best_action_onehots], dim=0)

        return HierarchicalPlanResult(
            phase1_result=phase1,
            phase2_result=phase2,
            combined_actions=combined_actions,
            combined_onehots=combined_onehots,
            is_cross_room=True,
            start_concept_code=start_code,
            goal_concept_code=goal_code,
            total_score=phase1.best_score + phase2.best_score,
        )
