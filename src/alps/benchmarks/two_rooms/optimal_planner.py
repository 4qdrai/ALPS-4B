"""
True BFS optimal planner / oracle for the Two Rooms env (baseline + complex).

The hand-coded heuristic "oracle" is unreliable (it oscillates at doors and can't
handle the key-gated 4-room layout), so it can't serve as a ceiling. This planner
does breadth-first search over the env's ACTUAL deterministic transitions in the
discretized state space (x, y, has_key), returning the shortest action sequence to
a goal-with-key state. It provides:
  - a valid success ceiling (~1.0 on any solvable config),
  - exact optimal path lengths for SPL,
  - clean optimal trajectories for training data.

Markovian only without momentum, so use hazards=False (the System-2 routing
benchmark). With ice/wind the state would also need velocity.
"""
from __future__ import annotations
from collections import deque
import numpy as np

from alps.benchmarks.two_rooms.environment import TwoRoomsEnv


def _key(x, y, hk, cell=0.1):
    return (round(float(x) / cell), round(float(y) / cell), int(hk))


def bfs_actions(env: TwoRoomsEnv, max_expand: int = 300000):
    """Return the optimal action list from the env's CURRENT reset state to a
    goal-with-key state, or None if unreachable. Mutates a scratch copy's pose."""
    target = env.target_pos.copy()
    key_pos = env.key_pos.copy() if env.complex_mode else None
    reach = 0.5
    # env.step() renders a 128x128 image via _get_obs every call — skip it during
    # search (BFS only needs position + has_key), or it's millions of renders.
    _orig_render = env.render
    env.render = lambda: np.zeros((1, 1, 3), dtype=np.uint8)

    start = (float(env.agent_pos[0]), float(env.agent_pos[1]), bool(getattr(env, "has_key", False)))
    came = {_key(*start): (None, None)}
    q = deque([start])
    goal_k = None
    while q and len(came) < max_expand:
        st = q.popleft()
        need_key = env.complex_mode
        if (not need_key or st[2]) and np.hypot(st[0] - target[0], st[1] - target[1]) < reach:
            goal_k = _key(*st); break
        for a in range(4):
            env.agent_pos = np.array([st[0], st[1]], dtype=np.float32)
            env.has_key = st[2]
            env.ice_momentum = np.zeros(2, dtype=np.float32)
            env.done = False
            if key_pos is not None:
                env.key_pos = key_pos.copy()
            o, _, _, info = env.step(a)
            ns = (float(o["position"][0]), float(o["position"][1]),
                  bool(info.get("has_key", st[2])))
            k = _key(*ns)
            if k not in came:
                came[k] = (_key(*st), a)
                q.append(ns)
    env.render = _orig_render
    if goal_k is None:
        return None
    acts = []
    k = goal_k
    while came[k][0] is not None:
        pk, a = came[k]
        acts.append(a); k = pk
    return acts[::-1]


def optimal_episode(seed: int, complex_mode: bool = True, hazards: bool = False,
                    max_steps: int = 300):
    """Plan (on a reset scratch env) + execute the optimal policy in an identical
    fresh env (same seed -> same layout). Returns (success, steps, plan)."""
    planv = TwoRoomsEnv(seed=seed, complex_mode=complex_mode, hazards=hazards); planv.reset()
    plan = bfs_actions(planv)
    if plan is None:
        return False, max_steps, None
    env = TwoRoomsEnv(seed=seed, complex_mode=complex_mode, hazards=hazards); env.reset()
    for i, a in enumerate(plan[:max_steps]):
        _, _, done, _ = env.step(int(a))
        if done:
            return True, i + 1, plan
    return False, len(plan), plan


def optimal_success_rate(n=60, complex_mode=True, hazards=False, base_seed=4000):
    ok, steps = [], []
    for i in range(n):
        s, st, _ = optimal_episode(base_seed + i, complex_mode, hazards)
        ok.append(s); steps.append(st)
    return float(np.mean(ok)), float(np.mean([t for t, o in zip(steps, ok) if o]) if any(ok) else 0)


if __name__ == "__main__":
    for hz in (False, True):
        r, s = optimal_success_rate(60, complex_mode=True, hazards=hz)
        print(f"complex hazards={hz}: BFS-optimal success {r:.2f} | avg steps {s:.0f}")
    r, s = optimal_success_rate(60, complex_mode=False, hazards=False)
    print(f"baseline: BFS-optimal success {r:.2f} | avg steps {s:.0f}")
