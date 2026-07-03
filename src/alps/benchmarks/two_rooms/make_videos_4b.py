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
    load_model, gather, HistoryBuffer, hist_greedy_action_latent, build_graph_raw, REACH,
    fit_ridge_decode, calibrate_bn, fit_softargmax_decode, fit_calibrated_softargmax,
    _gather_token_grids, gather_pred_grids)
from alps.evaluation.diagnose_control import fit_calibrated_decode
from alps.core.slot_readout import fit_slot_decode


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


@torch.no_grad()
def record_spatial(model, W, seed, sr, gr, device, decode_state, readout, graph,
                   strategy, complex_mode=False, size=288, ctrl_k=3, max_steps=160,
                   block_mode=False, block_wall=False, block_gate=False, decode_pred=None,
                   block_radius=None, block_step_scale=None):
    """SPATIAL-readout, predictor-DECODED control (mirrors validate_temporal.run_episode_spatial)
    while capturing env frames -- the path that ACTUALLY routes under pure SSL. The global-pool
    `record` above is position-blind (the small agent is diluted), so the Four-Brain panel would
    stall there even on a good model; here the trained op predictor rolls K steps, the spatial
    readout + frozen ridge (`decode_state`) reads the predicted agent POSITION, and the agent
    steers toward the position-faithful graph waypoint.
      operative : greedy to the decoded GOAL (System 1) -> stalls at the wall / locked door.
      fourbrain : follow the latent-graph waypoints (complex: routes through the KEY)."""
    _bkw = dict(block_mode=block_mode, block_wall=block_wall, block_gate=block_gate,
                block_radius=block_radius, block_step_scale=block_step_scale)
    env = TwoRoomsEnv(seed=seed, complex_mode=complex_mode, hazards=False, **_bkw)
    obs = env.reset() if (complex_mode or block_mode) else env.reset(start_room=sr, goal_room=gr)
    goal_xy = obs["target"].copy()
    buf = HistoryBuffer(model, W, device, readout=readout); buf.reset(obs_to_frame(obs, device))
    eg = TwoRoomsEnv(seed=seed, complex_mode=complex_mode, hazards=False, **_bkw)
    eg.reset() if (complex_mode or block_mode) else eg.reset(start_room=gr, goal_room=gr)
    eg.agent_pos = goal_xy.copy()
    if complex_mode or block_gate:
        eg.has_key = True   # goal latent = block at the target with the gate already open
    # decode_state (fit on REAL frames) reads the current/goal frame position; decode_pred
    # (fit on the op-predictor's OWN outputs -- the calibrated decode, 0.97 vs 0.66) reads the
    # IMAGINED next position in the rollout. Both return world units, so they compare directly.
    if decode_pred is None:
        decode_pred = decode_state
    goal_grid = model.encode_frame(obs_to_frame({"image": eg.render()}, device).unsqueeze(0))
    goal_pos = decode_state(goal_grid)[0].cpu().numpy()
    waypoints, wp = [goal_pos], 0
    if strategy == "fourbrain":
        sn = graph.node_of_latent(readout(buf.cur_z).squeeze(0).cpu().numpy())
        gn = graph.node_of_latent(readout(goal_grid).squeeze(0).cpu().numpy())
        path = graph.shortest_path(sn, gn) or [gn]
        seg = path[1:] if len(path) > 1 else path[:]
        waypoints = [graph.decoded_xy[n] for n in seg] + [goal_pos]
    wp_overlay = waypoints[:-1] if len(waypoints) > 1 else None
    label = "Operative (System 1)" if strategy == "operative" else "Four-Brain (System 2)"
    frames, solved = [], False
    for s in range(max_steps):
        sub = waypoints[min(wp, len(waypoints) - 1)]
        dec = decode_state(buf.cur_z)[0].cpu().numpy()                 # decoded pos overlay
        frames.append(_frame(obs["image"], size, label, waypoints=wp_overlay, dot=dec))
        best_a, best_d = 0, 1e30
        for a in range(4):
            pos_a = buf.rollout_decode(decode_pred, a, ctrl_k).cpu().numpy()
            d = float(np.linalg.norm(pos_a - sub))
            if d < best_d:
                best_d, best_a = d, a
        obs, _, done, info = env.step(best_a); buf.push(obs_to_frame(obs, device), best_a)
        if strategy == "fourbrain" and wp < len(waypoints) - 1:
            cur_pos = decode_state(buf.cur_z)[0].cpu().numpy()
            if np.linalg.norm(cur_pos - waypoints[wp]) < REACH:
                wp += 1
        reached = done if complex_mode else (done or info["distance"] < REACH)
        if reached:
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
    # CRITICAL: batch-dependent encoder BN -> single-frame control encoding is off-distribution
    # without this. calibrate_bn makes encode_frame(one frame) batch-independent. See encoders.py.
    if not getattr(args, "no_bn_calib", False):
        calibrate_bn(model, frames_t, device)
        print("[calibrate_bn] encoder BN running stats populated (single-frame inference fixed)")
    rng = np.random.RandomState(0); idx = rng.permutation(total)[: args.probe_samples]
    Z, _, _, _, P = gather(model, frames_t, positions, room_ids, idx, device)
    spatial = getattr(args, "spatial", False)
    if spatial:
        # SPATIAL readout + frozen ridge + predictor-decoded rollout control: the routing path
        # (the global pool is position-blind under pure SSL). Mirrors validate_temporal --spatial.
        g = args.spatial_grid
        readout = lambda z: model.spatial_readout(z, grid=g)

        @torch.no_grad()
        def _gather_readout(ix):
            out_f = []
            for c0 in range(0, len(ix), 128):
                b = torch.as_tensor(np.asarray(ix[c0:c0 + 128]))
                out_f.append(readout(model.encode_frame(frames_t[b].to(device).float() / 255.0)).cpu())
            return torch.cat(out_f)

        ridge_decode = fit_ridge_decode(_gather_readout(idx), P.float(), device)
        if getattr(args, "readout", "ridge") == "slot":
            # object-centric slot control decode (size-invariant; aggregates the diffuse
            # imagination): real-frame slots for current/goal, calibrated slots for the rollout.
            tr_s = idx[:min(len(idx), 6000)]
            sl = fit_slot_decode(_gather_token_grids(model, frames_t, tr_s, device),
                                 P.float()[:len(tr_s)], device, num_slots=6, epochs=3000)
            Zp, Yp = gather_pred_grids(model, frames_t, positions, actions, starts, total, W, device, n_win=6000)
            sl_c = fit_slot_decode(Zp, Yp, device, num_slots=6, epochs=3000)
            del Zp, Yp
            decode_state = sl
            decode_pred = sl_c
        elif getattr(args, "readout", "ridge") == "softargmax":
            # sub-cell heatmap centroid: sharp position for a SMALL agent among distractors.
            # decode_state reads REAL frames; decode_pred is CALIBRATED on the predictor's own
            # (off-manifold) outputs. The graph node positions still use the ridge (built in
            # readout space) -- only the per-step CONTROL decode switches to soft-argmax.
            decode_state = fit_softargmax_decode(model, frames_t, positions, idx, device)
            decode_pred = fit_calibrated_softargmax(model, frames_t, positions, actions, starts, total, W, device)
        else:
            decode_state = lambda grid: ridge_decode(model.spatial_readout(grid, grid=g))
            # CALIBRATED decode: a frozen ridge fit on the op-predictor's OWN outputs (imagined
            # next latent) -> removes the off-manifold linear distortion so the ROLLOUT reads the
            # imagined position accurately (bake-off: 0.97 vs 0.66 for the real-frame ridge).
            ridge_calib = fit_calibrated_decode(model, frames_t, positions, actions, starts,
                                                total, readout, W, device)
            decode_pred = lambda grid: ridge_calib(model.spatial_readout(grid, grid=g))
        graph = build_graph_raw(model, ridge_decode, frames_t, positions, room_ids, starts,
                                total, device, k=args.fine_k, S=max(1, args.stride // 2), readout=readout)
        graph.decoded_xy = ridge_decode(torch.tensor(graph.centroids, device=device)).cpu().numpy()

        def record_fn(seed, sr, gr, strategy):
            return record_spatial(model, W, seed, sr, gr, device, decode_state, readout, graph,
                                  strategy, complex_mode, args.size, getattr(args, "ctrl_k", 3),
                                  block_mode=getattr(args, "block_mode", False),
                                  block_wall=getattr(args, "block_wall", False),
                                  block_gate=getattr(args, "block_gate", False),
                                  block_radius=getattr(args, "block_radius", None),
                                  block_step_scale=getattr(args, "block_step_scale", None),
                                  decode_pred=decode_pred)
    else:
        torch.set_grad_enabled(True); decode_op = fit_probe(Z, P, device); torch.set_grad_enabled(False)
        graph, featurize = build_for_mode(model, decode_op, frames_t, actions, positions, room_ids,
                                          starts, total, device, complex_mode, None,
                                          args.coarse_k, args.fine_k, args.stride)

        def record_fn(seed, sr, gr, strategy):
            return record(model, W, seed, sr, gr, device, decode_op, graph, featurize,
                         strategy, complex_mode, args.size)
    tag = ("blockgate" if getattr(args, "block_gate", False) else
           "blockwall" if getattr(args, "block_wall", False) else "block") if getattr(args, "block_mode", False) \
          else ("complex" if complex_mode else "simple")
    # block_mode reset is random (opposite sides under block_wall), so sr/gr are ignored there.
    cfgs = [(0, 3, 2000 + i) for i in range(40)] if complex_mode else \
           [(0, 1, 1000 + i) for i in range(40)]  # cross-room / cross-wall configs
    made = 0
    for sr, gr, seed in cfgs:
        if made >= n_clips:
            break
        fb, ok_fb = record_fn(seed, sr, gr, "fourbrain")
        if not ok_fb and not args.include_failures:
            continue   # prefer clips where the Four-Brain actually solves (the proof)
        op, ok_op = record_fn(seed, sr, gr, "operative")
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
    ap.add_argument("--spatial", action="store_true",
                    help="Render with the SPATIAL readout + predictor-decoded rollout control "
                         "(grid x grid) -- the path that routes under pure SSL. WITHOUT this the "
                         "renderer uses the position-blind global pool and the Four-Brain panel "
                         "will stall even on a good model. Match the gate's --spatial-grid/--ctrl-k.")
    ap.add_argument("--spatial-grid", type=int, default=8)
    ap.add_argument("--ctrl-k", type=int, default=3)
    ap.add_argument("--block-mode", action="store_true",
                    help="render the Block-Rooms env (large decodable block) instead of the rooms env")
    ap.add_argument("--block-wall", action="store_true",
                    help="Block-Rooms WALL+GAP: greedy operative stalls at the wall, four-brain "
                         "routes through the gap (the hierarchy proof). Implies --block-mode.")
    ap.add_argument("--block-gate", action="store_true",
                    help="Block-Rooms SWITCH-GATE: key-locked gap. Greedy PROVABLY fails (never "
                         "fetches the key); only strategic key->gate->goal routing solves it. "
                         "The hierarchy-SUPREMACY proof. Implies --block-wall/--block-mode.")
    ap.add_argument("--no-bn-calib", action="store_true",
                    help="disable encoder BatchNorm running-stat calibration (debug only)")
    ap.add_argument("--block-radius", type=float, default=None, help="block render radius (MUST match training)")
    ap.add_argument("--block-step-scale", type=float, default=None, help="block step scale (MUST match training)")
    ap.add_argument("--readout", choices=["ridge", "softargmax", "slot"], default="ridge",
                    help="control decode: 'ridge' (grid-pool linear), 'softargmax' (sub-cell "
                         "centroid, sharp real-frame decode) or 'slot' (Slot-Attention object "
                         "binding -- size-invariant AND reads the diffuse imagination).")
    a = ap.parse_args()
    if a.block_gate:
        a.block_wall = True
    if a.block_wall:
        a.block_mode = True
    if a.block_mode:
        a.simple_only = True   # block-mode is a single env (no complex variant here)
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
