"""
Two Rooms Environment for ALPS-4B Benchmark.

A 10x10 continuous 2D grid world split into two rooms by a vertical wall
at x=5.0 with a 1.0-unit door gap centered at y=5.0. The agent (red dot)
must navigate to a target (green dot) that may be in the same or opposite room.

Actions are discrete 4-way:
    0 = up   (+y)
    1 = down (-y)
    2 = left (-x)
    3 = right(+x)

Rendering is pure numpy (no external graphics deps), producing 128x128 RGB frames.
"""

import numpy as np
from typing import Optional, Dict, Tuple, Any


class TwoRoomsEnv:
    """Continuous 2D navigation environment with two rooms and a door."""

    # --- World geometry -------------------------------------------------------
    WORLD_SIZE = 10.0
    WALL_X = 5.0                    # vertical wall position
    DOOR_Y_MIN = 4.5               # door gap lower bound
    DOOR_Y_MAX = 5.5               # door gap upper bound
    AGENT_RADIUS = 0.3             # collision radius for agent
    STEP_SIZE = 0.3                # movement per action
    BOUNDARY_MIN = 0.3             # = AGENT_RADIUS  (agent can't poke out)
    BOUNDARY_MAX = 9.7             # = WORLD_SIZE - AGENT_RADIUS

    # --- Rendering ------------------------------------------------------------
    RENDER_SIZE = 128              # output image resolution (square)
    COLOR_BACKGROUND = np.array([40, 40, 40], dtype=np.uint8)
    COLOR_FLOOR = np.array([200, 200, 200], dtype=np.uint8)
    COLOR_WALL = np.array([101, 67, 33], dtype=np.uint8)       # dark brown
    COLOR_AGENT = np.array([220, 50, 50], dtype=np.uint8)       # red
    COLOR_TARGET = np.array([50, 200, 50], dtype=np.uint8)      # green

    # Wall rendering thickness in world units
    WALL_THICKNESS = 0.15

    # --- Action mapping -------------------------------------------------------
    #   0=up(+y), 1=down(-y), 2=left(-x), 3=right(+x)
    ACTION_DELTAS = np.array([
        [0.0,  0.3],   # up
        [0.0, -0.3],   # down
        [-0.3, 0.0],   # left
        [0.3,  0.0],   # right
    ], dtype=np.float32)

    NUM_ACTIONS = 4

    def __init__(self, seed: Optional[int] = None):
        """
        Args:
            seed: optional RNG seed for reproducibility.
        """
        self.rng = np.random.RandomState(seed)

        # State
        self.agent_pos = np.array([2.5, 5.0], dtype=np.float32)
        self.target_pos = np.array([7.5, 5.0], dtype=np.float32)
        self.done = False
        self.steps = 0

        # Pre-compute pixel grid for rendering (world coords per pixel center)
        px = np.linspace(0, self.WORLD_SIZE, self.RENDER_SIZE, endpoint=False)
        px += (self.WORLD_SIZE / self.RENDER_SIZE) / 2.0  # pixel centers
        # x varies across columns, y varies across rows (top = high y)
        self._grid_x, self._grid_y = np.meshgrid(px, px[::-1])

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        start_room: Optional[int] = None,
        goal_room: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Reset the environment and return the initial observation.

        Args:
            start_room: 0 (left) or 1 (right). Random if None.
            goal_room:  0 (left) or 1 (right). If None, 50/50 same/other.

        Returns:
            obs dict with keys: 'image', 'position', 'target', 'room_id'.
        """
        self.done = False
        self.steps = 0

        # --- Agent start position ---
        if start_room is None:
            start_room = self.rng.randint(0, 2)
        self.agent_pos = self._random_position_in_room(start_room)

        # --- Target position ---
        if goal_room is None:
            # 50 % same room, 50 % other room
            if self.rng.rand() < 0.5:
                goal_room = start_room
            else:
                goal_room = 1 - start_room
        self.target_pos = self._random_position_in_room(goal_room)

        return self._get_obs()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute one discrete action.

        Args:
            action: integer in {0, 1, 2, 3}.

        Returns:
            (obs_dict, reward, done, info)
        """
        assert 0 <= action < self.NUM_ACTIONS, f"Invalid action {action}"

        if self.done:
            return self._get_obs(), 0.0, True, {"already_done": True}

        old_pos = self.agent_pos.copy()
        delta = self.ACTION_DELTAS[action]
        new_pos = old_pos + delta

        # --- Wall collision ---
        new_pos = self._apply_wall_collision(old_pos, new_pos)

        # --- Boundary clipping ---
        new_pos = np.clip(new_pos, self.BOUNDARY_MIN, self.BOUNDARY_MAX)

        self.agent_pos = new_pos
        self.steps += 1

        # --- Reward: negative distance to target ---
        dist = float(np.linalg.norm(self.agent_pos - self.target_pos))
        reward = -dist

        # --- Done: reached target within 0.5 units ---
        if dist < 0.5:
            self.done = True
            reward = 10.0  # bonus for reaching target

        info = {
            "distance": dist,
            "steps": self.steps,
            "room_id": self.get_room_id(self.agent_pos),
        }
        return self._get_obs(), reward, self.done, info

    def render(self) -> np.ndarray:
        """Render current state as a 128x128x3 uint8 RGB numpy array.

        Uses distance-field anti-aliased circles and filled rectangles.
        No external rendering libraries required.
        """
        img = np.full(
            (self.RENDER_SIZE, self.RENDER_SIZE, 3),
            self.COLOR_BACKGROUND,
            dtype=np.uint8,
        )

        gx = self._grid_x  # (128, 128)  world-x per pixel
        gy = self._grid_y  # (128, 128)  world-y per pixel

        # --- Draw room floors (two rectangles, leaving a thin wall gap) ---
        left_room = (gx >= 0) & (gx < self.WALL_X - self.WALL_THICKNESS)
        right_room = (gx > self.WALL_X + self.WALL_THICKNESS) & (gx <= self.WORLD_SIZE)
        floor_mask = left_room | right_room
        img[floor_mask] = self.COLOR_FLOOR

        # --- Draw wall (vertical bar at x=5, except door gap) ---
        wall_mask = (
            (np.abs(gx - self.WALL_X) <= self.WALL_THICKNESS)
            & ~((gy >= self.DOOR_Y_MIN) & (gy <= self.DOOR_Y_MAX))
        )
        img[wall_mask] = self.COLOR_WALL

        # Door region gets floor color
        door_mask = (
            (np.abs(gx - self.WALL_X) <= self.WALL_THICKNESS)
            & (gy >= self.DOOR_Y_MIN)
            & (gy <= self.DOOR_Y_MAX)
        )
        img[door_mask] = self.COLOR_FLOOR

        # --- Draw target (green circle, anti-aliased) ---
        img = self._draw_circle_aa(img, self.target_pos, self.AGENT_RADIUS, self.COLOR_TARGET)

        # --- Draw agent (red circle, anti-aliased, drawn on top) ---
        img = self._draw_circle_aa(img, self.agent_pos, self.AGENT_RADIUS, self.COLOR_AGENT)

        return img

    @staticmethod
    def get_room_id(pos: np.ndarray) -> int:
        """Return 0 if position is in the left room (x < 5), else 1."""
        return 0 if pos[0] < 5.0 else 1

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _random_position_in_room(self, room: int) -> np.ndarray:
        """Sample a random position inside the given room, respecting agent radius."""
        if room == 0:
            x = self.rng.uniform(self.BOUNDARY_MIN, self.WALL_X - self.AGENT_RADIUS)
        else:
            x = self.rng.uniform(self.WALL_X + self.AGENT_RADIUS, self.BOUNDARY_MAX)
        y = self.rng.uniform(self.BOUNDARY_MIN, self.BOUNDARY_MAX)
        return np.array([x, y], dtype=np.float32)

    def _apply_wall_collision(
        self, old_pos: np.ndarray, new_pos: np.ndarray
    ) -> np.ndarray:
        """If the agent tries to cross the wall outside the door, clip x.

        The wall is at x=WALL_X. The door gap is y ∈ [DOOR_Y_MIN, DOOR_Y_MAX].
        If the agent's y is NOT in the door gap and the movement crosses x=5,
        we clip x to the wall boundary (accounting for agent radius).
        """
        old_x, new_x = old_pos[0], new_pos[0]
        new_y = new_pos[1]

        # Check if movement crosses the wall
        crosses_wall = (old_x < self.WALL_X and new_x >= self.WALL_X) or \
                       (old_x > self.WALL_X and new_x <= self.WALL_X)
        # Also block if agent is right at the wall and pushing into it
        # (edge case: old_x == WALL_X is on the boundary)

        if not crosses_wall:
            # Even if not crossing, check proximity: if agent is near wall,
            # ensure it doesn't overlap the wall outside door
            in_door = self.DOOR_Y_MIN <= new_y <= self.DOOR_Y_MAX
            if not in_door:
                # Coming from left, don't let agent radius overlap wall
                if new_x > self.WALL_X - self.AGENT_RADIUS and old_x < self.WALL_X:
                    new_pos = new_pos.copy()
                    new_pos[0] = self.WALL_X - self.AGENT_RADIUS
                # Coming from right, don't let agent radius overlap wall
                elif new_x < self.WALL_X + self.AGENT_RADIUS and old_x > self.WALL_X:
                    new_pos = new_pos.copy()
                    new_pos[0] = self.WALL_X + self.AGENT_RADIUS
            return new_pos

        # Movement crosses the wall line
        in_door = self.DOOR_Y_MIN <= new_y <= self.DOOR_Y_MAX
        if in_door:
            # Allow passage through the door
            return new_pos

        # Block: clip x to the wall surface
        new_pos = new_pos.copy()
        if old_x < self.WALL_X:
            new_pos[0] = self.WALL_X - self.AGENT_RADIUS
        else:
            new_pos[0] = self.WALL_X + self.AGENT_RADIUS
        return new_pos

    def _draw_circle_aa(
        self,
        img: np.ndarray,
        center: np.ndarray,
        radius: float,
        color: np.ndarray,
    ) -> np.ndarray:
        """Draw a filled, anti-aliased circle using a distance field.

        For each pixel, compute distance from pixel center (in world coords)
        to circle center.  Pixels fully inside get the color; pixels on the
        edge get alpha-blended with a 1-pixel-wide (in world units) falloff.

        Args:
            img:    (H, W, 3) uint8 image to draw on (modified in-place).
            center: (2,) world coordinates [x, y].
            radius: circle radius in world units.
            color:  (3,) uint8 RGB color.

        Returns:
            The modified image.
        """
        cx, cy = float(center[0]), float(center[1])

        # Distance from every pixel center to circle center
        dx = self._grid_x - cx
        dy = self._grid_y - cy
        dist = np.sqrt(dx * dx + dy * dy)

        # Anti-aliasing: 1 pixel in world coords ≈ WORLD_SIZE / RENDER_SIZE
        pixel_size = self.WORLD_SIZE / self.RENDER_SIZE
        # Smooth step: alpha = 1 inside, 0 outside, smooth transition over ~1 pixel
        alpha = np.clip((radius - dist) / pixel_size + 0.5, 0.0, 1.0)

        # Only process pixels where alpha > 0
        mask = alpha > 0.0
        if not np.any(mask):
            return img

        # Blend: out = alpha * color + (1 - alpha) * background
        alpha_3d = alpha[mask, np.newaxis]  # (N, 1)
        blended = (
            alpha_3d * color.astype(np.float32)
            + (1.0 - alpha_3d) * img[mask].astype(np.float32)
        )
        img[mask] = np.clip(blended, 0, 255).astype(np.uint8)

        return img

    def _get_obs(self) -> Dict[str, Any]:
        """Build the observation dictionary."""
        return {
            "image": self.render(),                         # (128, 128, 3) uint8
            "position": self.agent_pos.copy(),              # (2,) float32
            "target": self.target_pos.copy(),               # (2,) float32
            "room_id": self.get_room_id(self.agent_pos),    # int 0 or 1
        }
