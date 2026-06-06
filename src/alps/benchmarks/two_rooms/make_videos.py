"""
Generate proof videos (GIFs) for the ALPS-4B Two Rooms validation:

  1. crossroom_edge.gif      — side-by-side, SAME cross-room config:
        left  = operative-greedy (goal-only)  -> stalls at the wall
        right = latent-graph strategic plan    -> routes through the door, reaches goal
     This is the visual proof of the hierarchy edge.

  2. decoder_overlay.gif     — a solved episode with the DECODED position (cyan)
        overlaid on the true agent (red): proof the latent is position-decodable.

  3. solved_crossroom_*.gif  — a few latent-graph episodes that solve cross-room.

Pure PIL (no ffmpeg/imageio dependency). Frames come from the env's native
renderer, upscaled and annotated.

USAGE
  PYTHONPATH=src python -m alps.benchmarks.two_rooms.make_videos \
      --model-path results/two_rooms/validation/repr_world_model_fs4.pt \
      --data-path data/two_rooms/trajectories_large.pt --limit-clips 1500
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, "src")

import argparse
import numpy as np
import torch
from PIL import Image, ImageDraw

from alps.benchmarks.two_rooms.dataset import TwoRoomsDataset
from alps.benchmarks.two_rooms.environment import TwoRoomsEnv
from alps.benchmarks.two_rooms.world_model_planning import (
    greedy_step_action, strategic_waypoints, obs_to_frame,
)
from alps.benchmarks.two_rooms.run_ablation_ladder import load_world_model
from alps.core.latent_graph import build_latent_graph
from alps.evaluation.repr_decoder_gate import gate_g1, split_dataset

WORLD = 10.0
REACH = 0.6


def _world_to_px(x, y, size):
    px = int(x / WORLD * size)
    py = int((WORLD - y) / WORLD * size)  # flip y (image origin top-left)
    return min(max(px, 0), size - 1), min(max(py, 0), size - 1)


def render_frame(img_uint8, size, lines, dot_xy=None, dot_color=(0, 255, 255)):
    """Upscale env frame, annotate text lines, optionally draw a decoded dot."""
    pil = Image.fromarray(img_uint8).resize((size, size), Image.NEAREST).convert("RGB")
    d = ImageDraw.Draw(pil)
    if dot_xy is not None:
        px, py = _world_to_px(dot_xy[0], dot_xy[1], size)
        r = max(3, size // 48)
        d.ellipse([px - r, py - r, px + r, py + r], outline=dot_color, width=2)
    y = 2
    for ln in lines:
        d.text((4, y), ln, fill=(255, 255, 255))
        y += 12
    return pil


def save_gif(frames, path, duration=90):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    print(f"[video] {path} ({len(frames)} frames)")


def run_record(env, obs, select_action, label, size, max_steps=130, decode_fn=None, model=None, device=None):
    """Run an episode, recording annotated frames. select_action(obs, step)->int."""
    frames, reached = [], False
    for step in range(max_steps):
        dot = None
        if decode_fn is not None:
            with torch.no_grad():
                z = model.encode_frame(obs_to_frame(obs, device).unsqueeze(0))
                dot = decode_fn(z)[0].cpu().numpy()
        status = "navigating"
        frames.append(render_frame(env.render(), size, [label, f"step {step}", status], dot_xy=dot))
        a = select_action(obs, step)
        obs, _, done, info = env.step(a)
        if done or info["distance"] < REACH:
            reached = True
            dot = None
            if decode_fn is not None:
                with torch.no_grad():
                    z = model.encode_frame(obs_to_frame(obs, device).unsqueeze(0))
                    dot = decode_fn(z)[0].cpu().numpy()
            frames.append(render_frame(env.render(), size, [label, f"step {step+1}", "SUCCESS"], dot_xy=dot))
            break
    if not reached:
        last = frames[-1]
        d = ImageDraw.Draw(last)
        d.text((4, size - 14), "STUCK / TIMEOUT", fill=(255, 80, 80))
    return frames, reached


def greedy_selector(model, decode_fn, goal_xy, device):
    return lambda obs, step: greedy_step_action(model, decode_fn, obs_to_frame(obs, device), goal_xy, device)


def waypoint_selector(model, decode_fn, waypoints, device):
    state = {"i": 0}

    def sel(obs, step):
        wp = waypoints[min(state["i"], len(waypoints) - 1)]
        if state["i"] < len(waypoints) - 1 and np.linalg.norm(obs["position"] - waypoints[state["i"]]) < REACH:
            state["i"] += 1
            wp = waypoints[min(state["i"], len(waypoints) - 1)]
        return greedy_step_action(model, decode_fn, obs_to_frame(obs, device), wp, device)
    return sel


def hstack(frames_a, frames_b, gap=8):
    n = max(len(frames_a), len(frames_b))
    fa = frames_a + [frames_a[-1]] * (n - len(frames_a))
    fb = frames_b + [frames_b[-1]] * (n - len(frames_b))
    out = []
    for a, b in zip(fa, fb):
        w = a.width + gap + b.width
        c = Image.new("RGB", (w, a.height), (20, 20, 20))
        c.paste(a, (0, 0)); c.paste(b, (a.width + gap, 0))
        out.append(c)
    return out


def graph_waypoints_for(model, decode_fn, graph, env, start_room, goal_room, seed, device):
    obs = env.reset(start_room=start_room, goal_room=goal_room)
    goal_xy = obs["target"].copy()
    with torch.no_grad():
        z_start = model.encode_frame(obs_to_frame(obs, device).unsqueeze(0)).mean(1).squeeze(0).cpu().numpy()
        eg = TwoRoomsEnv(seed=seed); eg.reset(start_room=goal_room, goal_room=goal_room)
        eg.agent_pos = goal_xy.copy()
        z_goal = model.encode_frame(obs_to_frame({"image": eg.render()}, device).unsqueeze(0)).mean(1).squeeze(0).cpu().numpy()
    wps = graph.waypoints(z_start, z_goal)
    wps.append(goal_xy.copy())
    return obs, wps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="results/two_rooms/validation/repr_world_model_fs4.pt")
    ap.add_argument("--data-path", default="data/two_rooms/trajectories_large.pt")
    ap.add_argument("--save-dir", default="results/two_rooms/videos")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--frame-skip", type=int, default=4)
    ap.add_argument("--limit-clips", type=int, default=1500)
    ap.add_argument("--graph-k", type=int, default=20)
    ap.add_argument("--size", type=int, default=320)
    args = ap.parse_args()

    device = torch.device(args.device)
    ds = TwoRoomsDataset(args.data_path, clip_length=8, stride=4, frame_skip=args.frame_skip)
    if args.limit_clips:
        ds.clip_indices = ds.clip_indices[: args.limit_clips]
    tr, va = split_dataset(ds, val_frac=0.2, seed=0)

    model = load_world_model(args.model_path, device)
    g1, decode_fn = gate_g1(model, ds, tr, va, device, probe_epochs=80)
    print(f"[decoder] held-out error {g1['held_out_decode_error_world_units']:.3f} wu")
    graph = build_latent_graph(model, decode_fn, ds, device, k=args.graph_k)

    # ---- 1. Side-by-side cross-room edge: find a seed where graph solves it ----
    chosen = None
    for seed in range(40):
        env = TwoRoomsEnv(seed=seed)
        obs_g, wps = graph_waypoints_for(model, decode_fn, graph, env, 0, 1, seed, device)
        fr_g, ok_g = run_record(env, obs_g, waypoint_selector(model, decode_fn, wps, device),
                                "Latent-graph (System 2)", args.size)
        if ok_g:
            chosen = seed
            env_o = TwoRoomsEnv(seed=seed)
            obs_o = env_o.reset(start_room=0, goal_room=1)
            goal_xy = obs_o["target"].copy()
            fr_o, ok_o = run_record(env_o, obs_o, greedy_selector(model, decode_fn, goal_xy, device),
                                    "Operative-only (greedy)", args.size)
            save_gif(hstack(fr_o, fr_g), os.path.join(args.save_dir, "crossroom_edge.gif"))
            print(f"[edge] seed={seed} operative_success={ok_o} graph_success={ok_g}")
            break
    if chosen is None:
        print("[warn] no cross-room success found in 40 seeds; skipping edge video")

    # ---- 2. Decoder overlay on a (solved) episode ----
    env = TwoRoomsEnv(seed=chosen if chosen is not None else 1)
    obs_g, wps = graph_waypoints_for(model, decode_fn, graph, env, 0, 1,
                                     chosen if chosen is not None else 1, device)
    fr_dec, _ = run_record(env, obs_g, waypoint_selector(model, decode_fn, wps, device),
                           "Decoded (cyan) vs true (red)", args.size,
                           decode_fn=decode_fn, model=model, device=device)
    save_gif(fr_dec, os.path.join(args.save_dir, "decoder_overlay.gif"))

    # ---- 3. A few solved cross-room episodes ----
    saved = 0
    for seed in range(40):
        if saved >= 3:
            break
        env = TwoRoomsEnv(seed=100 + seed)
        obs_g, wps = graph_waypoints_for(model, decode_fn, graph, env, 0, 1, 100 + seed, device)
        fr, ok = run_record(env, obs_g, waypoint_selector(model, decode_fn, wps, device),
                            "Latent-graph cross-room", args.size)
        if ok:
            save_gif(fr, os.path.join(args.save_dir, f"solved_crossroom_{saved}.gif"))
            saved += 1
    print(f"[done] videos in {args.save_dir}")


if __name__ == "__main__":
    main()
