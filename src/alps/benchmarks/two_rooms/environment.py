"""
Two Rooms Environment for ALPS-4B Benchmark.

Supports two modes:
1. Baseline Mode: A 10x10 continuous 2D grid world split into two rooms by a vertical wall
   at x=5.0 with a 1.0-unit door gap centered at y=5.0. Discrete 4-way actions.
2. Complex Mode: A 4-room quadrant maze with locked vertical door, randomized visual keys,
   slippery ice zones (momentum sliding), windy storm zones (drift force), dynamic background
   visual static noise, and safe havens. Perfect testbed for advanced Strategic, Tactical,
   and Fallback layers.

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
    """Continuous 2D navigation environment supporting Baseline and Complex modes."""

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
    COLOR_KEY = np.array([255, 215, 0], dtype=np.uint8)         # bright yellow

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

    def __init__(self, seed: Optional[int] = None, complex_mode: bool = False,
                 hazards: bool = True):
        """
        Args:
            seed: optional RNG seed for reproducibility.
            complex_mode: whether to enable the 4-room key-gated complex navigation mode.
            hazards: in complex mode, apply ice/wind momentum (System-1 control
                challenge). Set False to test key-gated routing (System-2) in isolation.
        """
        self.complex_mode = complex_mode
        self.hazards = hazards
        self.rng = np.random.RandomState(seed)

        # State
        self.agent_pos = np.array([2.5, 5.0], dtype=np.float32)
        self.target_pos = np.array([7.5, 5.0], dtype=np.float32)
        self.done = False
        self.steps = 0

        # Complex Mode specific state
        self.key_pos = np.array([2.5, 7.5], dtype=np.float32) # Default inside Room 1 (top-left)
        self.has_key = False
        self.ice_momentum = np.array([0.0, 0.0], dtype=np.float32)

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
        """Reset the environment and return the initial observation."""
        self.done = False
        self.steps = 0
        self.has_key = False
        self.ice_momentum = np.array([0.0, 0.0], dtype=np.float32)

        if not self.complex_mode:
            # --- Baseline Reset ---
            if start_room is None:
                start_room = self.rng.randint(0, 2)
            self.agent_pos = self._random_position_in_room(start_room)

            if goal_room is None:
                if self.rng.rand() < 0.5:
                    goal_room = start_room
                else:
                    goal_room = 1 - start_room
            self.target_pos = self._random_position_in_room(goal_room)
        else:
            # --- Complex Mode Reset ---
            # Agent always starts in Room 0 (bottom-left)
            self.agent_pos = self._random_position_in_room(0)
            # Goal is always in Room 3 (bottom-right)
            self.target_pos = self._random_position_in_room(3)
            # Key spawns in Room 1 (top-left) or Room 2 (top-right)
            key_room = self.rng.choice([1, 2])
            self.key_pos = self._random_position_in_room(key_room)

        return self._get_obs()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute one discrete action."""
        assert 0 <= action < self.NUM_ACTIONS, f"Invalid action {action}"

        if self.done:
            return self._get_obs(), 0.0, True, {"already_done": True}

        old_pos = self.agent_pos.copy()
        delta = self.ACTION_DELTAS[action].copy()

        # --- Apply Variable Physics (Complex Mode Only) ---
        # Hazards (ice/wind) are a System-1 control challenge; disable them to test
        # the System-2 key-gated ROUTING challenge in isolation (self.hazards=False).
        if self.complex_mode and self.hazards:
            current_room = self.get_room_id(old_pos, self.complex_mode)

            # Room 1: Slippery Ice (Low friction momentum slide)
            if current_room == 1:
                # delta = 40% action direction + 60% previous sliding momentum
                delta = 0.4 * delta + 0.6 * self.ice_momentum
                self.ice_momentum = delta.copy()
            else:
                self.ice_momentum = np.array([0.0, 0.0], dtype=np.float32)

            # Room 2: Windy Storm (Constant southward draft drift)
            if current_room == 2:
                delta[1] -= 0.15

        new_pos = old_pos + delta

        # --- Wall collision & boundaries ---
        new_pos = self._apply_wall_collision(old_pos, new_pos)
        new_pos = np.clip(new_pos, self.BOUNDARY_MIN, self.BOUNDARY_MAX)

        self.agent_pos = new_pos
        self.steps += 1

        # --- Key Pickup Check (Complex Mode Only) ---
        if self.complex_mode and not self.has_key:
            dist_to_key = np.linalg.norm(self.agent_pos - self.key_pos)
            if dist_to_key < (self.AGENT_RADIUS + 0.25):
                self.has_key = True

        # --- Reward ---
        dist = float(np.linalg.norm(self.agent_pos - self.target_pos))
        reward = -dist

        # Done Check: reached target within 0.5 units
        # In complex mode, we ALSO enforce that the agent must have the key to unlock victory!
        if dist < 0.5:
            if not self.complex_mode or self.has_key:
                self.done = True
                reward = 10.0  # bonus

        info = {
            "distance": dist,
            "steps": self.steps,
            "room_id": self.get_room_id(self.agent_pos, self.complex_mode),
        }
        if self.complex_mode:
            info["has_key"] = self.has_key
            info["key_distance"] = float(np.linalg.norm(self.agent_pos - self.key_pos))

        return self._get_obs(), reward, self.done, info

    def render(self) -> np.ndarray:
        """Render current state as a 128x128x3 uint8 RGB numpy array."""
        img = np.full(
            (self.RENDER_SIZE, self.RENDER_SIZE, 3),
            self.COLOR_BACKGROUND,
            dtype=np.uint8,
        )

        gx = self._grid_x
        gy = self._grid_y

        # --- 1. Draw floors ---
        if not self.complex_mode:
            left_room = (gx >= 0) & (gx < self.WALL_X - self.WALL_THICKNESS)
            right_room = (gx > self.WALL_X + self.WALL_THICKNESS) & (gx <= self.WORLD_SIZE)
            floor_mask = left_room | right_room
            img[floor_mask] = self.COLOR_FLOOR
        else:
            # 4 quadrant floors
            q0 = (gx < 5.0 - self.WALL_THICKNESS) & (gy < 5.0 - self.WALL_THICKNESS)
            q1 = (gx < 5.0 - self.WALL_THICKNESS) & (gy > 5.0 + self.WALL_THICKNESS)
            q2 = (gx > 5.0 + self.WALL_THICKNESS) & (gy > 5.0 + self.WALL_THICKNESS)
            q3 = (gx > 5.0 + self.WALL_THICKNESS) & (gy < 5.0 - self.WALL_THICKNESS)
            floor_mask = q0 | q1 | q2 | q3
            img[floor_mask] = self.COLOR_FLOOR

        # --- 2. Draw walls ---
        if not self.complex_mode:
            wall_mask = (
                (np.abs(gx - self.WALL_X) <= self.WALL_THICKNESS)
                & ~((gy >= self.DOOR_Y_MIN) & (gy <= self.DOOR_Y_MAX))
            )
            img[wall_mask] = self.COLOR_WALL

            # Door gap
            door_mask = (
                (np.abs(gx - self.WALL_X) <= self.WALL_THICKNESS)
                & (gy >= self.DOOR_Y_MIN)
                & (gy <= self.DOOR_Y_MAX)
            )
            img[door_mask] = self.COLOR_FLOOR
        else:
            # Vertical wall at x=5
            v_wall = np.abs(gx - 5.0) <= self.WALL_THICKNESS
            # Lock is unlocked (gets floor color) if has_key is True or one-way upper door is open from left (gx < 5.0 and gy >= 5.0)
            v_door = (4.5 <= gy) & (gy <= 5.5) & (self.has_key or ((gx < 5.0) & (gy >= 5.0)))

            # Horizontal wall at y=5
            h_wall = np.abs(gy - 5.0) <= self.WALL_THICKNESS
            # Horizontal doors are always open at x in [2, 3] and [7, 8]
            h_door = ((2.0 <= gx) & (gx <= 3.0)) | ((7.0 <= gx) & (gx <= 8.0))

            wall_mask = (v_wall & ~v_door) | (h_wall & ~h_door)
            img[wall_mask] = self.COLOR_WALL

        # --- 3. Draw dynamic static noise in background (Complex Mode distractors) ---
        if self.complex_mode:
            # 1.5% chance of visual noise blocks on floor
            noise_mask = (self.rng.rand(*gx.shape) < 0.015) & floor_mask
            noise_val = self.rng.randint(160, 220, (np.sum(noise_mask), 3), dtype=np.uint8)
            img[noise_mask] = noise_val

        # --- 4. Draw Key (yellow circle, Complex Mode only) ---
        if self.complex_mode and not self.has_key:
            img = self._draw_circle_aa(img, self.key_pos, 0.25, self.COLOR_KEY)

        # --- 5. Draw target (green circle) ---
        img = self._draw_circle_aa(img, self.target_pos, self.AGENT_RADIUS, self.COLOR_TARGET)

        # --- 6. Draw agent (red circle, on top) ---
        img = self._draw_circle_aa(img, self.agent_pos, self.AGENT_RADIUS, self.COLOR_AGENT)

        return img

    @staticmethod
    def get_room_id(pos: np.ndarray, complex_mode: bool = False) -> int:
        """Return room index based on position."""
        if not complex_mode:
            return 0 if pos[0] < 5.0 else 1
        else:
            # Quadrants mapping
            if pos[0] < 5.0:
                return 0 if pos[1] < 5.0 else 1  # 0: bottom-left, 1: top-left (Ice)
            else:
                return 2 if pos[1] >= 5.0 else 3 # 2: top-right (Wind), 3: bottom-right (Goal)

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _random_position_in_room(self, room: int) -> np.ndarray:
        """Sample a random position inside the given room, respecting boundaries."""
        if not self.complex_mode:
            if room == 0:
                x = self.rng.uniform(self.BOUNDARY_MIN, self.WALL_X - self.AGENT_RADIUS)
            else:
                x = self.rng.uniform(self.WALL_X + self.AGENT_RADIUS, self.BOUNDARY_MAX)
            y = self.rng.uniform(self.BOUNDARY_MIN, self.BOUNDARY_MAX)
            return np.array([x, y], dtype=np.float32)
        else:
            margin = 0.5
            if room == 0:
                x = self.rng.uniform(self.BOUNDARY_MIN + margin, 5.0 - margin)
                y = self.rng.uniform(self.BOUNDARY_MIN + margin, 5.0 - margin)
            elif room == 1:
                x = self.rng.uniform(self.BOUNDARY_MIN + margin, 5.0 - margin)
                y = self.rng.uniform(5.0 + margin, self.BOUNDARY_MAX - margin)
            elif room == 2:
                x = self.rng.uniform(5.0 + margin, self.BOUNDARY_MAX - margin)
                y = self.rng.uniform(5.0 + margin, self.BOUNDARY_MAX - margin)
            else:
                x = self.rng.uniform(5.0 + margin, self.BOUNDARY_MAX - margin)
                y = self.rng.uniform(self.BOUNDARY_MIN + margin, 5.0 - margin)
            return np.array([x, y], dtype=np.float32)

    def _apply_wall_collision(
        self, old_pos: np.ndarray, new_pos: np.ndarray
    ) -> np.ndarray:
        """Enforce physical wall borders."""
        if not self.complex_mode:
            # Baseline vertical wall
            old_x, new_x = old_pos[0], new_pos[0]
            new_y = new_pos[1]

            crosses_wall = (old_x < self.WALL_X and new_x >= self.WALL_X) or \
                           (old_x > self.WALL_X and new_x <= self.WALL_X)

            if not crosses_wall:
                in_door = self.DOOR_Y_MIN <= new_y <= self.DOOR_Y_MAX
                if not in_door:
                    if new_x > self.WALL_X - self.AGENT_RADIUS and old_x < self.WALL_X:
                        new_pos = new_pos.copy()
                        new_pos[0] = self.WALL_X - self.AGENT_RADIUS
                    elif new_x < self.WALL_X + self.AGENT_RADIUS and old_x > self.WALL_X:
                        new_pos = new_pos.copy()
                        new_pos[0] = self.WALL_X + self.AGENT_RADIUS
                return new_pos

            in_door = self.DOOR_Y_MIN <= new_y <= self.DOOR_Y_MAX
            if in_door:
                return new_pos

            new_pos = new_pos.copy()
            if old_x < self.WALL_X:
                new_pos[0] = self.WALL_X - self.AGENT_RADIUS
            else:
                new_pos[0] = self.WALL_X + self.AGENT_RADIUS
            return new_pos
        else:
            # Complex mode checking (4 chambers)
            old_x, old_y = old_pos[0], old_pos[1]
            new_x, new_y = new_pos[0], new_pos[1]

            # --- Check 1: Vertical Wall at x=5 ---
            crosses_v_wall = (old_x < 5.0 and new_x >= 5.0) or (old_x > 5.0 and new_x <= 5.0)
            # Upper half of vertical door (y >= 5.0) is one-way open from left (Room 1 -> 2) without key.
            # Lower half (y < 5.0) is strictly locked, requiring the key (Room 0 -> 3).
            is_upper_left = (5.0 <= new_y <= 5.5) and (old_x < 5.0)
            in_v_door = (4.5 <= new_y <= 5.5) and (self.has_key or is_upper_left)

            if not crosses_v_wall:
                if not in_v_door:
                    if new_x > 5.0 - self.AGENT_RADIUS and old_x < 5.0:
                        new_x = 5.0 - self.AGENT_RADIUS
                    elif new_x < 5.0 + self.AGENT_RADIUS and old_x > 5.0:
                        new_x = 5.0 + self.AGENT_RADIUS
            else:
                if not in_v_door:
                    new_x = 5.0 - self.AGENT_RADIUS if old_x < 5.0 else 5.0 + self.AGENT_RADIUS

            # --- Check 2: Horizontal Wall at y=5 ---
            crosses_h_wall = (old_y < 5.0 and new_y >= 5.0) or (old_y > 5.0 and new_y <= 5.0)
            in_h_door = (2.0 <= new_x <= 3.0) or (7.0 <= new_x <= 8.0)

            if not crosses_h_wall:
                if not in_h_door:
                    if new_y > 5.0 - self.AGENT_RADIUS and old_y < 5.0:
                        new_y = 5.0 - self.AGENT_RADIUS
                    elif new_y < 5.0 + self.AGENT_RADIUS and old_y > 5.0:
                        new_y = 5.0 + self.AGENT_RADIUS
            else:
                if not in_h_door:
                    new_y = 5.0 - self.AGENT_RADIUS if old_y < 5.0 else 5.0 + self.AGENT_RADIUS

            return np.array([new_x, new_y], dtype=np.float32)

    def _draw_circle_aa(
        self,
        img: np.ndarray,
        center: np.ndarray,
        radius: float,
        color: np.ndarray,
    ) -> np.ndarray:
        """Draw an anti-aliased circle using a distance field."""
        cx, cy = float(center[0]), float(center[1])

        dx = self._grid_x - cx
        dy = self._grid_y - cy
        dist = np.sqrt(dx * dx + dy * dy)

        pixel_size = self.WORLD_SIZE / self.RENDER_SIZE
        alpha = np.clip((radius - dist) / pixel_size + 0.5, 0.0, 1.0)

        mask = alpha > 0.0
        if not np.any(mask):
            return img

        alpha_3d = alpha[mask, np.newaxis]
        blended = (
            alpha_3d * color.astype(np.float32)
            + (1.0 - alpha_3d) * img[mask].astype(np.float32)
        )
        img[mask] = np.clip(blended, 0, 255).astype(np.uint8)

        return img

    def _get_obs(self) -> Dict[str, Any]:
        """Build the observation dictionary."""
        obs = {
            "image": self.render(),
            "position": self.agent_pos.copy(),
            "target": self.target_pos.copy(),
            "room_id": self.get_room_id(self.agent_pos, self.complex_mode),
        }
        if self.complex_mode:
            obs["has_key"] = float(self.has_key)
            obs["key_pos"] = self.key_pos.copy()
        return obs
