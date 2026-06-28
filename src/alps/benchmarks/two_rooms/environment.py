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
    # Agent/target/key are drawn LARGER than their physics radius so the mover occupies enough
    # pixels for the pure-SSL objective to ENCODE its position. With a ~2-4px agent (radius 0.3)
    # the dot is a negligible fraction of every 16x16 patch, the next-frame prediction is
    # dominated by static background, and the latent stays position-blind -> predictor-decoded
    # control fails (the tiny-agent-in-static-scene pathology). Enlarging it is RENDER-ONLY:
    # AGENT_RADIUS (collision), STEP_SIZE, boundaries and the door geometry are unchanged, so the
    # navigation task is identical; only the visible blob grows enough to be encodable/decodable.
    AGENT_RENDER_RADIUS = 0.8      # ~20px diameter @128 ~= 1.25 patches @ patch16 (encodable)
    TARGET_RENDER_RADIUS = 0.7
    KEY_RENDER_RADIUS = 0.6

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
                 hazards: bool = True, egocentric: bool = False, perception_radius: float = None):
        """
        Args:
            seed: optional RNG seed for reproducibility.
            complex_mode: whether to enable the 4-room key-gated complex navigation mode.
            hazards: in complex mode, apply ice/wind momentum (System-1 control
                challenge). Set False to test key-gated routing (System-2) in isolation.
            egocentric: render the world AGENT-CENTERED (the agent sits at the frame centre
                and the room/door/target scroll around it as it moves). This makes EVERY pixel
                action-dependent -- the next frame is the current one shifted by the action --
                so the pure-SSL next-frame predictor is forced to learn controllable dynamics
                (the top-down god-view leaves the agent a ~2% static-background fraction, which
                is why the predictor ignores the action). Physics/dynamics are unchanged; only
                the camera changes. This is the real-video-aligned observation model.
        """
        self.complex_mode = complex_mode
        self.hazards = hazards
        self.egocentric = egocentric
        # Limited perception (egocentric only): the agent observes only a disk of this radius (wu)
        # around itself; everything beyond is unobserved (background). This makes the far static
        # structure trivial-to-predict, so the ONLY non-trivial thing to predict is the local
        # optical flow == the consequence of the action -> the predictor must learn it. Far goals
        # are then reached by chaining local moves toward graph waypoints + latent-RAG memory.
        self.perception_radius = perception_radius
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

        # Position-locked floor texture (FIXED across all env instances = one shared world).
        # Egocentric views over open floor are otherwise featureless -> position unrecoverable;
        # a non-repeating textured floor that scrolls with the agent gives optical-flow / absolute
        # position cues EVERYWHERE. Used only in egocentric mode (absolute keeps the plain floor).
        # Sharp non-repeating tiled texture (16x16 random colours). High spatial frequency =>
        # sharp position decode + a strong motion cue for inverse dynamics. The predictor doesn't
        # need to imagine the texture pixel-perfectly: inverse dynamics disentangles a smooth,
        # low-dim POSITION factor (action-displacement), which is what the predictor forecasts.
        self._FLOOR_TEX_RES = 16
        _trng = np.random.RandomState(12345)
        self._floor_tex = _trng.randint(75, 215, (self._FLOOR_TEX_RES, self._FLOOR_TEX_RES, 3),
                                         dtype=np.uint8)

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

    def _ego(self, p):
        """World position -> position in the current FRAME. Identity in absolute mode;
        agent-centred (p - agent + centre) in egocentric mode."""
        if not self.egocentric:
            return p
        return p - self.agent_pos + self.WORLD_SIZE / 2.0

    def render(self) -> np.ndarray:
        """Render current state as a 128x128x3 uint8 RGB numpy array."""
        img = np.full(
            (self.RENDER_SIZE, self.RENDER_SIZE, 3),
            self.COLOR_BACKGROUND,
            dtype=np.uint8,
        )

        gx = self._grid_x
        gy = self._grid_y
        if self.egocentric:
            # world coord at each pixel = agent_pos + (frame_coord - centre); pixels mapping
            # outside [0, WORLD_SIZE] fall through to background. The whole scene scrolls with the
            # agent, so the NEXT frame is the current one shifted by the action.
            gx = self._grid_x + (self.agent_pos[0] - self.WORLD_SIZE / 2.0)
            gy = self._grid_y + (self.agent_pos[1] - self.WORLD_SIZE / 2.0)

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

        # Position-locked textured floor (egocentric only): scrolls with the agent so every
        # open-floor view still encodes absolute position (fixes the featureless white-room).
        if self.egocentric:
            R = self._FLOOR_TEX_RES
            ix = np.clip((gx / self.WORLD_SIZE * R).astype(np.int64), 0, R - 1)
            iy = np.clip((gy / self.WORLD_SIZE * R).astype(np.int64), 0, R - 1)
            tex = self._floor_tex[iy, ix]
            img[floor_mask] = tex[floor_mask]

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
            img = self._draw_circle_aa(img, self._ego(self.key_pos), self.KEY_RENDER_RADIUS, self.COLOR_KEY)

        # --- 5. Draw target (green circle) ---
        img = self._draw_circle_aa(img, self._ego(self.target_pos), self.TARGET_RENDER_RADIUS, self.COLOR_TARGET)

        # --- 6. Draw agent (red circle, on top; at frame centre in egocentric mode) ---
        img = self._draw_circle_aa(img, self._ego(self.agent_pos), self.AGENT_RENDER_RADIUS, self.COLOR_AGENT)

        # --- 7. Limited perception: mask everything beyond the perception radius (egocentric) ---
        if self.egocentric and self.perception_radius is not None:
            c = self.WORLD_SIZE / 2.0   # agent is at the frame centre in egocentric mode
            dist_from_agent = np.sqrt((self._grid_x - c) ** 2 + (self._grid_y - c) ** 2)
            img[dist_from_agent > self.perception_radius] = self.COLOR_BACKGROUND

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


# ══════════════════════════════════════════════════════════════════════════════
# H6 — N-room environment (WS-F)
# ══════════════════════════════════════════════════════════════════════════════

class NRoomsEnv:
    """N-room maze for H6 (edge-grows-with-horizon).

    Geometry: N rooms in a horizontal chain.  Room k occupies x ∈ [k·W, (k+1)·W].
    Between adjacent rooms k and k+1 there is a vertical wall with a door gap at
    y ∈ [4.5, 5.5]. There are no hazards (pure routing challenge: the agent must
    pass through N-1 doors in sequence to reach the goal room N-1).

    Why H6 is interesting: for N=2 a greedy policy sometimes gets lucky; for N≥4
    the probability of accidentally threading all doors drops to near-zero and
    the hierarchical graph planner is the only reliable strategy.

    API mirrors TwoRoomsEnv (reset / step / render / get_room_id).
    """

    WORLD_SIZE = 10.0
    RENDER_SIZE = 128
    AGENT_RADIUS = 0.3
    STEP_SIZE = 0.3
    BOUNDARY_MIN = 0.3
    BOUNDARY_MAX = 9.7
    WALL_THICKNESS = 0.15
    DOOR_Y_MIN = 4.5
    DOOR_Y_MAX = 5.5
    NUM_ACTIONS = 4

    ACTION_DELTAS = np.array([[0.0, 0.3], [0.0, -0.3], [-0.3, 0.0], [0.3, 0.0]],
                              dtype=np.float32)

    # Room colour palette — unique colour per room (makes the task visually readable)
    ROOM_COLORS = [
        np.array([200, 200, 200], dtype=np.uint8),   # room 0 — light grey
        np.array([200, 220, 255], dtype=np.uint8),   # room 1 — blue tint
        np.array([220, 255, 200], dtype=np.uint8),   # room 2 — green tint
        np.array([255, 220, 200], dtype=np.uint8),   # room 3 — orange tint
        np.array([255, 200, 220], dtype=np.uint8),   # room 4 — pink tint
        np.array([220, 200, 255], dtype=np.uint8),   # room 5 — purple tint
        np.array([255, 255, 180], dtype=np.uint8),   # room 6 — yellow tint
        np.array([180, 255, 255], dtype=np.uint8),   # room 7 — cyan tint
    ]
    COLOR_BACKGROUND = np.array([40, 40, 40], dtype=np.uint8)
    COLOR_WALL = np.array([101, 67, 33], dtype=np.uint8)
    COLOR_AGENT = np.array([220, 50, 50], dtype=np.uint8)
    COLOR_TARGET = np.array([50, 200, 50], dtype=np.uint8)

    def __init__(self, n_rooms: int = 4, seed: Optional[int] = None):
        assert 2 <= n_rooms <= 8, "n_rooms must be 2–8"
        self.n_rooms = n_rooms
        self.room_width = self.WORLD_SIZE / n_rooms          # width of each room
        self.wall_xs = [self.room_width * k for k in range(1, n_rooms)]  # wall x positions
        self.rng = np.random.RandomState(seed)

        self.agent_pos = np.array([self.room_width * 0.5, 5.0], dtype=np.float32)
        self.target_pos = np.array([self.WORLD_SIZE - self.room_width * 0.5, 5.0], dtype=np.float32)
        self.done = False; self.steps = 0

        px = np.linspace(0, self.WORLD_SIZE, self.RENDER_SIZE, endpoint=False)
        px += (self.WORLD_SIZE / self.RENDER_SIZE) / 2.0
        self._grid_x, self._grid_y = np.meshgrid(px, px[::-1])

    def get_room_id(self, pos: np.ndarray, *args) -> int:
        return int(min(self.n_rooms - 1, max(0, int(pos[0] / self.room_width))))

    def reset(self, start_room: int = 0, goal_room: Optional[int] = None) -> Dict[str, Any]:
        self.done = False; self.steps = 0
        if goal_room is None:
            goal_room = self.n_rooms - 1
        lo = start_room * self.room_width + self.AGENT_RADIUS + 0.2
        hi = (start_room + 1) * self.room_width - self.AGENT_RADIUS - 0.2
        self.agent_pos = np.array([
            self.rng.uniform(lo, hi),
            self.rng.uniform(self.BOUNDARY_MIN, self.BOUNDARY_MAX)
        ], dtype=np.float32)
        glo = goal_room * self.room_width + self.AGENT_RADIUS + 0.2
        ghi = (goal_room + 1) * self.room_width - self.AGENT_RADIUS - 0.2
        self.target_pos = np.array([
            self.rng.uniform(glo, ghi),
            self.rng.uniform(self.BOUNDARY_MIN, self.BOUNDARY_MAX)
        ], dtype=np.float32)
        return self._get_obs()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        delta = self.ACTION_DELTAS[action].copy()
        new_pos = np.clip(self.agent_pos + delta, self.BOUNDARY_MIN, self.BOUNDARY_MAX)
        new_pos = self._apply_all_walls(self.agent_pos, new_pos)
        self.agent_pos = new_pos; self.steps += 1
        dist = float(np.linalg.norm(self.agent_pos - self.target_pos))
        if dist < 0.5:
            self.done = True
        return self._get_obs(), -dist, self.done, {
            "distance": dist, "steps": self.steps,
            "room_id": self.get_room_id(self.agent_pos)}

    def _apply_all_walls(self, old_pos: np.ndarray, new_pos: np.ndarray) -> np.ndarray:
        """Apply collision for every vertical wall in the chain."""
        p = new_pos.copy()
        for wx in self.wall_xs:
            crosses = (old_pos[0] < wx and p[0] >= wx) or (old_pos[0] > wx and p[0] <= wx)
            in_door = self.DOOR_Y_MIN <= p[1] <= self.DOOR_Y_MAX
            if not crosses:
                if not in_door:
                    if p[0] > wx - self.AGENT_RADIUS and old_pos[0] < wx:
                        p = p.copy(); p[0] = wx - self.AGENT_RADIUS
                    elif p[0] < wx + self.AGENT_RADIUS and old_pos[0] > wx:
                        p = p.copy(); p[0] = wx + self.AGENT_RADIUS
            else:
                if not in_door:
                    p = p.copy()
                    p[0] = wx - self.AGENT_RADIUS if old_pos[0] < wx else wx + self.AGENT_RADIUS
        return p

    def render(self) -> np.ndarray:
        img = np.full((self.RENDER_SIZE, self.RENDER_SIZE, 3), self.COLOR_BACKGROUND, dtype=np.uint8)
        gx, gy = self._grid_x, self._grid_y
        # Draw room floors (distinct colours)
        for k in range(self.n_rooms):
            lo, hi = k * self.room_width, (k + 1) * self.room_width
            col = self.ROOM_COLORS[k % len(self.ROOM_COLORS)]
            img[(gx >= lo) & (gx < hi)] = col
        # Draw walls (with door gaps)
        for wx in self.wall_xs:
            wall = (np.abs(gx - wx) <= self.WALL_THICKNESS) & \
                   ~((gy >= self.DOOR_Y_MIN) & (gy <= self.DOOR_Y_MAX))
            img[wall] = self.COLOR_WALL
            door = (np.abs(gx - wx) <= self.WALL_THICKNESS) & \
                   (gy >= self.DOOR_Y_MIN) & (gy <= self.DOOR_Y_MAX)
            # colour the door gap with the left room's colour
            room_k = int(wx / self.room_width) - 1
            img[door] = self.ROOM_COLORS[max(0, room_k) % len(self.ROOM_COLORS)]
        # Agent and target
        img = self._draw_circle(img, self.target_pos, self.AGENT_RADIUS, self.COLOR_TARGET)
        img = self._draw_circle(img, self.agent_pos, self.AGENT_RADIUS, self.COLOR_AGENT)
        return img

    def _draw_circle(self, img, center, radius, color):
        cx, cy = float(center[0]), float(center[1])
        dist = np.sqrt((self._grid_x - cx) ** 2 + (self._grid_y - cy) ** 2)
        ps = self.WORLD_SIZE / self.RENDER_SIZE
        alpha = np.clip((radius - dist) / ps + 0.5, 0.0, 1.0)
        m = alpha > 0.0
        img[m] = np.clip(
            alpha[m, None] * color.astype(np.float32) + (1 - alpha[m, None]) * img[m].astype(np.float32),
            0, 255).astype(np.uint8)
        return img

    def _get_obs(self) -> Dict[str, Any]:
        return {"image": self.render(), "position": self.agent_pos.copy(),
                "target": self.target_pos.copy(),
                "room_id": self.get_room_id(self.agent_pos)}
