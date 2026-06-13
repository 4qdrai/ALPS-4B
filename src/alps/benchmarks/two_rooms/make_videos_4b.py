"""
Four-Brain PROOF VIDEOS on the Two-Rooms task (SIMPLE + COMPLEX), rendered from the
TEMPORAL hierarchy model (works on the UNSUPERVISED --lewm-ssl model too).

Each clip is side-by-side:
    LEFT  = Operative only (System 1, greedy to goal)  -> stalls at the wall / locked door
    RIGHT = Four-Brain (System 2: latent-graph plan + tactical sub-goals) -> threads the
            door, (complex) fetches the key, reaches the goal.
Overlays: planned latent-graph waypoints, the live decoded position, and a SOLVED/STALLED
badge. Records the env's OWN rendered frames as each controller acts.

Honesty: this renders the model's ACTUAL behaviour. The RIGHT panel "solves" only if the
trained model actually solves -- i.e. it is a real proof, gated on the model's quality
(read G1 first). On an unsupervised model that has not yet reached identifiability the
clip will faithfully show that too.

PIL for GIF; ffmpeg (PIL-frames -> PNG -> image2) for MP4 to avoid the GIF-demuxer
first-frame drop.

USAGE
  PYTHONPATH=src python -m alps.benchmarks.two_rooms.make_videos_4b \
      --model-path results/two_rooms/validation/unsupervised/unsup_temporal.pt \
      --data-path  data/two_rooms/trajectories_unsup.pt \
      --complex-model-path results/two_rooms/validation/unsup_temporal_complex.pt \
      --complex-data-path  data/two_rooms/trajectories_unsup_complex.pt \
      --save-dir results/two_rooms/videos_unsup
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, "src")
import argparse, shutil, subprocess, tempfile
import numpy as np
import torch
from PIL import Image, ImageDraw

from alps.benchmarks.two_rooms.environment import TwoRoomsEnv
from alps.benchmarks.two_rooms.world_model_planning import obs_to_frame
from alps.training.train_hier import load_raw
from alps.evaluation.validate_hierarchy import fit_probe
from alps.evaluation.validate_temporal import (
    load_model, gather, HistoryBuffer, hist_greedy_action_latent, build_graph_raw, REACH)


def _px(xy, size):
    return int(xy[0] / 10.0 * size), int((1 - xy[1] / 10.0) * size)


def _frame(img_uint8, size, label, waypoints=None, dot=None, solved=None):
    im = Image.fromarray(img_uint8).resize((size, size), Image.NEAREST).convert("RGB")
    d = ImageDraw.Draw(im)
    if waypoints is not None:
        for wp in waypoints:
            x, y = _px(wp, size)
            d.ellipse([x - 3, y - 3, x + 3, y + 3], outline=(0, 200, 255), width=2)
    if dot is not None:
        x, y = _px(dot, size)
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 255, 0))
    d.rectangle([0, 0, size - 1, 13], fill=(0, 0, 0))
    d.text((3, 2), label, fill=(255, 255, 255))
    if solved is not None:
        col = (0, 220, 0) if solved else (220, 60, 60)
        d.rectangle([0, 0, size - 1, size - 1], outline=col, width=3)
        d.text((size - 64, 2), "SOLVED" if solved else "STALLED", fill=col)
    return im


def _hstack(a, b, gap=10):
    n = min(len(a), len(b))
    a = a + [a[-1]] * (n - len(a)); b = b + [b[-1]] * (n - len(b))
    out = []
    for i in range(n):
        fa, fb = a[i], b[i]
        w, h = fa.width + gap + fb.width, max(fa.height, fb.height)
        im = Image.new("RGB", (w, h), (20, 20, 20))
        im.paste(fa, (0, 0)); im.paste(fb, (fa.width + gap, 0))
        out.append(im)
    return out


def _save(frames, path, fps=12):
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    print(f"[video] {path}")
    if shutil.which("ffmpeg") is None:
        print("[video] ffmpeg not found; GIF only"); return
    mp4 = path[:-4] + ".mp4"
    with tempfile.TemporaryDirectory() as td:
        for i, f in enumerate(frames):
            f.save(os.path.join(td, f"{i:04d}.png"))
        try:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                            "-i", os.path.join(td, "%04d.png"),
                            "-pix_fmt", "yuv420p",
                            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", mp4], check=True)
            print(f"[video] {mp4}")
        except Exception as e:
            print(f"[video] mp4 skipped: {e}")


@torch.no_grad()
def record(model, W, seed, sr, gr, device, decode_op, graph, featurize, strategy,
           complex_mode=False, size=288, max_steps=160):
    """LABEL-FREE control (latent-space), capturing env frames per step. `decode_op` is
    used ONLY for the on-screen position overlay (visualization), never for control."""
    env = TwoRoomsEnv(seed=seed, complex_mode=complex_mode, hazards=False)
    obs = env.reset() if complex_mode else env.reset(start_room=sr, goal_room=gr)
    goal_xy = obs["target"].copy()
    buf = HistoryBuffer(model, W, device); buf.reset(obs_to_frame(obs, device))
    eg = TwoRoomsEnv(seed=seed, complex_mode=complex_mode, hazards=False)
    eg.reset() if complex_mode else eg.reset(start_room=gr, goal_room=gr)
    eg.agent_pos = goal_xy.copy()
    if complex_mode:
        eg.has_key = True
    goal_lat = model.pool(model.encode_frame(obs_to_frame({"image": eg.render()}, device).unsqueeze(0))).squeeze(0)
    cents = torch.tensor(graph.centroids, device=device, dtype=torch.float32)
    seg_nodes, lat_waypoints, wp = [], None, 0
    if strategy == "fourbrain":
        sn = graph.node_of_latent(model.pool(buf.cur_z).squeeze(0).cpu().numpy())
        gn = graph.node_of_latent(goal_lat.cpu().numpy())
        path = graph.shortest_path(sn, gn) or [gn]
        seg_nodes = path[1:] if len(path) > 1 else path[:]
        lat_waypoints = [cents[n] for n in seg_nodes] + [goal_lat]
    # waypoint POSITIONS are decoded only for the overlay (measurement, not control)
    wp_overlay = [decode_op(cents[n].unsqueeze(0))[0].cpu().numpy() for n in seg_nodes] if seg_nodes else None
    label = "Operative (System 1)" if strategy == "operative" else "Four-Brain (System 2)"
    frames, solved = [], False
    for s in range(max_steps):
        sub_lat = goal_lat if strategy == "operative" else lat_waypoints[min(wp, len(lat_waypoints) - 1)]
        dec = decode_op(model.pool(buf.cur_z))[0].cpu().numpy()        # overlay only
        frames.append(_frame(obs["image"], size, label, waypoints=wp_overlay, dot=dec))
        a = hist_greedy_action_latent(buf, sub_lat, device)
        obs, _, done, info = env.step(a); buf.push(obs_to_frame(obs, device), a)
        if seg_nodes and wp < len(seg_nodes):
            cur = model.pool(buf.cur_z).squeeze(0)
            if int((cents - cur).norm(dim=1).argmin()) == seg_nodes[wp]:
                wp += 1
        if done if complex_mode else (done or info["distance"] < REACH):
            solved = True
            frames.append(_frame(obs["image"], size, label, waypoints=wp_overlay,
                                 dot=obs["position"], solved=True))
            break
    if not solved:
        frames[-1] = _frame(obs["image"], size, label, waypoints=wp_overlay,
                            dot=obs["position"], solved=False)
    return frames, solved


def build_for_mode(model, decode_op, frames_t, actions, positions, room_ids, starts, total,
                   device, complex_mode, hk, coarse_k, fine_k, stride):
    # LABEL-FREE: pure latent transition graph for both simple and complex. decode_op is
    # passed only so node centroids carry a decoded-xy for the overlay (not for control).
    graph = build_graph_raw(model, decode_op, frames_t, positions, room_ids, starts,
                            total, device, k=fine_k, S=max(1, stride // 2))
    featurize = lambda z: z
    return graph, featurize


def make_clips(model_path, data_path, complex_mode, save_dir, device, args, n_clips=3):
    model, W = load_model(model_path, device)
    frames_t, actions, positions, room_ids, starts = load_raw(data_path)
    total = frames_t.shape[0]
    hk = None
    rng = np.random.RandomState(0); idx = rng.permutation(total)[: args.probe_samples]
    Z, _, _, _, P = gather(model, frames_t, positions, room_ids, idx, device)
    torch.set_grad_enabled(True); decode_op = fit_probe(Z, P, device); torch.set_grad_enabled(False)
    graph, featurize = build_for_mode(model, decode_op, frames_t, actions, positions, room_ids,
                                      starts, total, device, complex_mode, hk,
                                      args.coarse_k, args.fine_k, args.stride)
    tag = "complex" if complex_mode else "simple"
    cfgs = [(0, 3, 2000 + i) for i in range(40)] if complex_mode else \
           [(0, 1, 1000 + i) for i in range(40)]  # cross-room configs
    made = 0
    for sr, gr, seed in cfgs:
        if made >= n_clips:
            break
        fb, ok_fb = record(model, W, seed, sr, gr, device, decode_op, graph, featurize,
                           "fourbrain", complex_mode, args.size)
        if not ok_fb and not args.include_failures:
            continue   # prefer clips where the Four-Brain actually solves (the proof)
        op, ok_op = record(model, W, seed, sr, gr, device, decode_op, graph, featurize,
                           "operative", complex_mode, args.size)
        out = _hstack(op, fb)
        path = os.path.join(save_dir, f"fourbrain_{tag}_seed{seed}.gif")
        _save(out, path, fps=args.fps)
        print(f"  [{tag} seed{seed}] operative {'SOLVED' if ok_op else 'stalled'} | "
              f"four-brain {'SOLVED' if ok_fb else 'stalled'}")
        made += 1
    if made == 0:
        print(f"  [{tag}] no Four-Brain successes to film (model likely undertrained; "
              f"use --include-failures to film attempts).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="results/two_rooms/validation/unsupervised/unsup_temporal.pt")
    ap.add_argument("--data-path", default="data/two_rooms/trajectories_unsup.pt")
    ap.add_argument("--complex-model-path", default="results/two_rooms/validation/unsupervised/unsup_temporal_complex.pt")
    ap.add_argument("--complex-data-path", default="data/two_rooms/trajectories_unsup_complex.pt")
    ap.add_argument("--save-dir", default="results/two_rooms/videos_unsup")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--probe-samples", type=int, default=4000)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--coarse-k", type=int, default=8)
    ap.add_argument("--fine-k", type=int, default=24)
    ap.add_argument("--size", type=int, default=288)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--n-clips", type=int, default=3)
    ap.add_argument("--include-failures", action="store_true")
    ap.add_argument("--simple-only", action="store_true")
    ap.add_argument("--complex-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.save_dir, exist_ok=True)
    dev = torch.device(a.device)
    if not a.complex_only:
        print("=== SIMPLE two-rooms (cross-room) ===")
        make_clips(a.model_path, a.data_path, False, a.save_dir, dev, a, a.n_clips)
    if not a.simple_only and os.path.exists(a.complex_model_path):
        print("=== COMPLEX two-rooms (key -> door -> goal) ===")
        make_clips(a.complex_model_path, a.complex_data_path, True, a.save_dir, dev, a, a.n_clips)


if __name__ == "__main__":
    main()
