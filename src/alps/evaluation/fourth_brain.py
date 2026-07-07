"""
THE FOURTH BRAIN — self-monitoring -> escalation -> fallback (gates H8/H9/H10).

Minimal honest form of the RSRA-4B loop (github.com/4qdrai/RSRA-4B) on the frozen
ALPS latent stack. Three LABEL-FREE monitors, computed every control step:

  m1  surprise          ||pool(op_predict_next(hist, a)) - pool(encode(actual))||
                        (the same signal as the Latent-RAG write trigger)
  m2  off-manifold      min distance of the pooled latent to the latent-graph
                        landmark centroids (far from everything we modeled)
  m3  progress stall    decoded-position displacement over the last `stall_w`
                        steps (raw value; alarm when it drops below eps)

Alarm(t) = [m1>th1] + [m2>th2] + [m3<eps] ;  alarmed if >= alarm_k of 3.
A PERSISTENT alarm (>= `patience` consecutive alarmed steps) escalates the tier:

  tier 0  OPERATIVE   greedy straight to goal (System 1, no plan)
  tier 1  TACTICAL    plan fine-latent-graph waypoints from the current state
  tier 2  STRATEGIC   full REPLAN from the current node (fresh route)
  tier 3  FALLBACK    route to the SAFE landmark (highest-visitation node =
                      best-modeled region), halt there: failed-but-SAFE.

Protocol (mirrors docs/EDGE_PROGRAM.md WS-I):
  PASS A  calibrate th1/th2/eps on a CALIBRATION seed split (fixed tactical
          policy, successful episodes' monitor quantiles). Thresholds frozen.
  PASS B  H8 on HELD-OUT seeds: do monitors predict failure? (AUROC, lead time)
  PASS C  H9: escalation (fallback recorded, not executed) vs fixed tiers;
          false-trigger rate = would-trigger among episodes that SUCCEEDED.
  PASS D  H10: fallback executed on trigger -> safe-state reach rate, vs a
          random-walk-after-trigger baseline on the same seeds.

USAGE
  PYTHONPATH=src python -m alps.evaluation.fourth_brain \
      --model-path results/two_rooms/validation/temporal_world_model.pt \
      --data-path data/two_rooms/trajectories.pt --n-cal 24 --n-eval 24
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, "src")
import argparse, json
import numpy as np
import torch
import torch.nn.functional as F

from alps.benchmarks.two_rooms.environment import TwoRoomsEnv
from alps.benchmarks.two_rooms.world_model_planning import obs_to_frame
from alps.training.train_hier import load_raw
from alps.evaluation.validate_hierarchy import fit_probe
from alps.evaluation.validate_temporal import (
    load_model, load_has_keys, gather, HistoryBuffer, hist_greedy_action_latent,
    build_graph_raw, build_graph_semantic, make_featurize, fit_key_probe, REACH,
    detect_key_pickups_unsup, gate_h4_key_detector, fit_ridge_decode)

TIERS = ("operative", "tactical", "strategic", "fallback")


# ---------------- helpers ----------------
_ENV_KW = {}          # Block-Rooms variant kwargs, set by run() from CLI (must match training)

_TIER_STYLE = [("OPERATIVE", (40, 130, 60)), ("TACTICAL", (40, 90, 170)),
               ("STRATEGIC", (190, 120, 30)), ("FALLBACK", (170, 40, 40))]


def _annotate_fb_frame(img, tier, m1, m2, m3, thr, alarmed, fb_active):
    """Render one monitored-episode frame with the SELF-MONITOR overlay: the active tier
    (operative->tactical->strategic->fallback), the 3 label-free monitors (green=ok, red=firing
    vs the calibrated threshold), and an ALARM/FALLBACK banner. This is the driving-safety story
    made visible: the imagination-error monitor spikes on unfamiliar dynamics -> the stack
    escalates -> falls back to the safe state."""
    from PIL import Image, ImageDraw
    name, tcol = _TIER_STYLE[min(tier, 3)]
    up = Image.fromarray(np.asarray(img, dtype=np.uint8)).resize((256, 256), Image.NEAREST)
    canvas = Image.new("RGB", (256, 256 + 48), (18, 18, 18))
    canvas.paste(up, (0, 48))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, 255, 47], fill=tcol)
    d.text((5, 2), f"TIER: {'FALLBACK' if fb_active else name}", fill=(255, 255, 255))
    ok, fire = (150, 220, 150), (255, 90, 90)
    if thr is not None:
        d.text((5, 18), f"m1 surprise {m1:5.2f}", fill=fire if m1 > thr["m1"] else ok)
        d.text((5, 31), f"m2 offmanif {m2:5.2f}", fill=fire if m2 > thr["m2"] else ok)
        d.text((140, 18), f"m3 stall {m3:5.2f}", fill=fire if m3 < thr["m3"] else ok)
    if alarmed:
        d.text((140, 31), "!! ALARM", fill=(255, 60, 60))
    return np.array(canvas)


def _annotate_titled(img, title, tcol, badge=None):
    """Generic titled frame: a colored header strip + optional badge (used by the RAG
    before/after video). Keeps the two panels' frame size identical for side-by-side stacking."""
    from PIL import Image, ImageDraw
    up = Image.fromarray(np.asarray(img, dtype=np.uint8)).resize((256, 256), Image.NEAREST)
    canvas = Image.new("RGB", (256, 256 + 30), (18, 18, 18))
    canvas.paste(up, (0, 30))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, 255, 29], fill=tcol)
    d.text((5, 2), title, fill=(255, 255, 255))
    if badge:
        d.text((5, 15), badge, fill=(255, 235, 90))
    return np.array(canvas)


def _save_sidebyside(framesL, framesR, path, fps=8):
    """Pad two annotated-frame lists to equal length and save a side-by-side gif (before|after)."""
    from PIL import Image
    n = max(len(framesL), len(framesR))
    def pad(fr):
        return fr + [fr[-1]] * (n - len(fr)) if fr else [np.zeros((286, 256, 3), np.uint8)] * n
    L, R = pad(list(framesL)), pad(list(framesR))
    ims = [Image.fromarray(np.concatenate([a, np.full((a.shape[0], 6, 3), 18, np.uint8), b], axis=1))
           for a, b in zip(L, R)]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0)


@torch.no_grad()
def predicted_pooled(model, buf, a, device, readout=None):
    """The readout latent the operative predictor expects after action `a` (pool by
    default; spatial readout in --spatial mode)."""
    readout = readout if readout is not None else model.pool
    z_hist = torch.stack(buf.z, dim=1)                                   # [1,W,N,D]
    a_idx = buf.a[1:] + [a]
    a_hist = F.one_hot(torch.tensor(a_idx, device=device), 4).float().unsqueeze(0)
    return readout(model.op_predict_next(z_hist, a_hist)).squeeze(0)     # [R]


@torch.no_grad()
def _goal_grid(model, seed, gr, goal_xy, complex_mode, device):
    """Render the goal IMAGE (agent at goal, key held in complex) -> encoded grid."""
    eg = TwoRoomsEnv(seed=seed, complex_mode=complex_mode, hazards=False, **_ENV_KW)
    eg.reset() if complex_mode else eg.reset(start_room=gr, goal_room=gr)
    eg.agent_pos = goal_xy.copy()
    if complex_mode:
        eg.has_key = True
    return model.encode_frame(obs_to_frame({"image": eg.render()}, device).unsqueeze(0))


@torch.no_grad()
def goal_latent(model, seed, gr, goal_xy, complex_mode, device, readout=None):
    """Goal IMAGE -> goal readout latent (no labels)."""
    readout = readout if readout is not None else model.pool
    return readout(_goal_grid(model, seed, gr, goal_xy, complex_mode, device)).squeeze(0)


@torch.no_grad()
def plan_path(model, graph, buf, goal_lat, device, readout=None):
    """LABEL-FREE latent-graph plan: node indices from current node -> goal node (start
    skipped). The latent transition graph routes through key-acquisition on its own."""
    readout = readout if readout is not None else model.pool
    sn = graph.node_of_latent(readout(buf.cur_z).squeeze(0).cpu().numpy())
    gn = graph.node_of_latent(goal_lat.cpu().numpy())
    path = graph.shortest_path(sn, gn) or [gn]
    return path[1:] if len(path) > 1 else path[:]


@torch.no_grad()
def plan_waypoints(model, graph, featurize, buf, goal_xy, seed, gr, complex_mode, has_key,
                   device, readout=None):
    """LABEL-FREE decoded waypoint POSITIONS along the start->goal latent-graph path
    (+ the goal). Used by the RAG control loop."""
    readout = readout if readout is not None else model.pool
    zg = readout(_goal_grid(model, seed, gr, goal_xy, complex_mode, device)).squeeze(0).cpu().numpy()
    zs = readout(buf.cur_z).squeeze(0).cpu().numpy()
    return list(graph.waypoints(zs, zg)) + [goal_xy.copy()]


def safe_node_of(graph):
    """Highest-visitation landmark = the best-modeled region (the safe state)."""
    return int((graph.edges.sum(1) + graph.edges.sum(0)).argmax())


def auroc(scores, labels):
    """Rank-based AUROC of `scores` for predicting labels==1 (no sklearn)."""
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---------------- the monitored episode ----------------
@torch.no_grad()
def run_episode_fb(model, W, seed, sr, gr, device, graph, ZC, policy, thr=None,
                   complex_mode=False, max_steps=140, alarm_k=2, patience=4, grace=6,
                   stall_w=8, fallback="record", readout=None, decode_state=None, frames_out=None):
    """LABEL-FREE control. policy in {operative, tactical, escalation}. Monitors are all
    on the readout latent: m1 surprise (pred vs actual), m2 off-manifold (dist to nearest
    landmark), m3 stall (readout displacement over a window). fallback: 'record' note
    would-trigger; 'on' route to the safe node; 'random' baseline. When decode_state is
    given (--spatial), control is PREDICTOR-DECODED on the spatial readout (action chosen
    by the predicted decoded POSITION) instead of latent nearest-neighbour."""
    readout = readout if readout is not None else model.pool
    env = TwoRoomsEnv(seed=seed, complex_mode=complex_mode, hazards=False, **_ENV_KW)
    obs = env.reset() if complex_mode else env.reset(start_room=sr, goal_room=gr)
    goal_xy = obs["target"].copy()
    buf = HistoryBuffer(model, W, device, readout=readout); buf.reset(obs_to_frame(obs, device))
    goal_grid = _goal_grid(model, seed, gr, goal_xy, complex_mode, device)
    goal_lat = readout(goal_grid).squeeze(0)
    goal_pos = decode_state(goal_grid)[0].cpu().numpy() if decode_state is not None else None
    cents = torch.tensor(graph.centroids, device=device, dtype=torch.float32)
    safe = safe_node_of(graph)

    tier = 0 if policy in ("operative", "escalation") else 1
    seg, wp = ([], 0)
    if tier == 1:
        seg = plan_path(model, graph, buf, goal_lat, device, readout)

    m1s, m2s, m3s, alarms, tiers_used = [], [], [], [], {0}
    consec, since_esc = 0, grace
    lat_hist = [readout(buf.cur_z).squeeze(0)]
    fb_would, fb_active, fb_seg, fb_wp = None, False, None, 0

    def near_node(za):
        return int((cents - za).norm(dim=1).argmin())

    for s in range(max_steps):
        # target NODE for this tier/fallback (None == steer to the goal)
        if fb_active and fallback == "on":
            tgt = fb_seg[min(fb_wp, len(fb_seg) - 1)] if fb_seg else safe
        elif tier == 0:
            tgt = None
        else:
            tgt = seg[min(wp, len(seg) - 1)] if wp < len(seg) else None
        # act: random-walk fallback baseline / predictor-decoded (spatial) / latent nearest
        if fb_active and fallback == "random":
            a = int(np.random.RandomState(seed * 7919 + s).randint(0, 4))
        elif decode_state is not None:
            sub_pos = goal_pos if tgt is None else graph.decoded_xy[tgt]
            a, best_d = 0, 1e30
            for cand in range(4):
                d = float(np.linalg.norm(buf.decode_next_for_action(decode_state, cand).cpu().numpy() - sub_pos))
                if d < best_d:
                    best_d, a = d, cand
        else:
            sub_lat = goal_lat if tgt is None else cents[tgt]
            a = hist_greedy_action_latent(buf, sub_lat, device)

        zp = predicted_pooled(model, buf, a, device, readout)
        obs, _, done, info = env.step(a); buf.push(obs_to_frame(obs, device), a)
        za = readout(buf.cur_z).squeeze(0); lat_hist.append(za)
        m1 = float((zp - za).norm())
        m2 = float(torch.cdist(za.unsqueeze(0), ZC).min())
        m3 = float((lat_hist[-1] - lat_hist[-stall_w]).norm()) if len(lat_hist) > stall_w else float("inf")
        m1s.append(m1); m2s.append(m2); m3s.append(m3)

        alarmed = False
        if thr is not None:
            score = int(m1 > thr["m1"]) + int(m2 > thr["m2"]) + int(m3 < thr["m3"])
            alarmed = score >= alarm_k
        alarms.append(bool(alarmed))
        consec = consec + 1 if alarmed else 0
        since_esc += 1
        cur = near_node(za)
        d_safe = float((za - cents[safe]).norm())

        if fb_active and fallback == "on":
            if fb_seg and fb_wp < len(fb_seg) - 1 and cur == fb_seg[fb_wp]:
                fb_wp += 1
            if cur == safe:
                return _result(False, s + 1, sr, gr, complex_mode, m1s, m2s, m3s, alarms,
                               fb_would, True, True, d_safe, tiers_used)
        elif seg and wp < len(seg) and cur == seg[wp]:
            wp += 1

        if policy == "escalation" and thr is not None and consec >= patience and since_esc >= grace \
                and not fb_active:
            consec, since_esc = 0, 0
            if tier < 2:
                tier += 1; tiers_used.add(tier)
                seg = plan_path(model, graph, buf, goal_lat, device); wp = 0
            else:
                if fb_would is None:
                    fb_would = s
                if fallback in ("on", "random"):
                    fb_active = True; tiers_used.add(3)
                    if fallback == "on":
                        pth = graph.shortest_path(cur, safe) or [safe]
                        fb_seg, fb_wp = (pth[1:] if len(pth) > 1 else pth), 0

        if frames_out is not None:
            cur_tier = 3 if fb_active else tier
            frames_out.append(_annotate_fb_frame(obs["image"], cur_tier, m1, m2, m3, thr,
                                                  alarmed, fb_active))

        reached = done if complex_mode else (done or info["distance"] < REACH)
        if reached and not fb_active:
            return _result(True, s + 1, sr, gr, complex_mode, m1s, m2s, m3s, alarms,
                           fb_would, False, False, d_safe, tiers_used)
    return _result(False, max_steps, sr, gr, complex_mode, m1s, m2s, m3s, alarms,
                   fb_would, fb_active, bool(near_node(za) == safe),
                   float((za - cents[safe]).norm()), tiers_used)


def _result(success, steps, sr, gr, cx, m1s, m2s, m3s, alarms, fb_would, fb_active,
            safe_reached, dist_safe, tiers_used):
    return {"success": success, "steps": steps, "is_cross": True if cx else sr != gr,
            "m1": m1s, "m2": m2s, "m3": m3s, "alarms": alarms,
            "fb_would_step": fb_would, "fb_executed": fb_active,
            "safe_reached": safe_reached, "dist_safe_end": dist_safe,
            "tiers_used": sorted(tiers_used)}


@torch.no_grad()
def run_episode_rag(model, W, seed, sr, gr, device, decode_op, graph, featurize, m1_thr,
                    mode, complex_mode=False, max_steps=140, readout=None,
                    frames_out=None, title="", tcol=(60, 60, 70)):
    """Latent-RAG in the control loop, gated by the SELF-MONITOR surprise signal m1.
      mode 'experience' : act; when surprise m1 > thr (the monitor fires), WRITE memory
                          key=context-latent, value=correction (actual_next - predicted).
      mode 'recall'     : act, but first RETRIEVE a correction for the current context
                          and APPLY it to the operative prediction (surprise-gated by the
                          RAG similarity threshold) -> memory steers control.
    Returns success (1/0)."""
    env = TwoRoomsEnv(seed=seed, complex_mode=complex_mode, hazards=False, **_ENV_KW)
    obs = env.reset() if complex_mode else env.reset(start_room=sr, goal_room=gr)
    goal_xy = obs["target"].copy()
    readout = readout if readout is not None else model.pool
    buf = HistoryBuffer(model, W, device, readout=readout); buf.reset(obs_to_frame(obs, device))
    waypoints = plan_waypoints(model, graph, featurize, buf, goal_xy, seed, gr,
                               complex_mode, bool(obs.get("has_key", 0)), device, readout)
    wp = 0
    for s in range(max_steps):
        sub = torch.tensor(waypoints[min(wp, len(waypoints) - 1)], device=device, dtype=torch.float32)
        ctx = buf.cur_z                                         # context BEFORE the step
        z_hist = torch.stack(buf.z, dim=1)
        best_a, best_d, best_zn = 0, 1e30, None
        for a in range(4):
            a_idx = buf.a[1:] + [a]
            a_hist = F.one_hot(torch.tensor(a_idx, device=device), 4).float().unsqueeze(0)
            z_next = model.op_predict_next(z_hist, a_hist)
            if mode == "recall":
                z_next = model.rag_correct(z_next, ctx)         # d_model correction
            d = float((decode_op(z_next)[0] - sub).norm())
            if d < best_d:
                best_d, best_a, best_zn = d, a, z_next
        # surprise m1 on the READOUT (matches the calibrated threshold); the LatentRAG
        # corrects the operative latent, so its key/value live in d_model -> use pool().
        best_pred = readout(best_zn).squeeze(0)
        obs, _, done, info = env.step(best_a); buf.push(obs_to_frame(obs, device), best_a)
        za = readout(buf.cur_z).squeeze(0)
        m1 = float((best_pred - za).norm())
        wrote = False
        if mode == "experience" and m1 > m1_thr:               # monitor fires -> memorize
            model.rag.write_memory(model.pool(ctx).squeeze(0),
                                   model.pool(buf.cur_z).squeeze(0) - model.pool(best_zn).squeeze(0))
            wrote = True
        if frames_out is not None:
            # only badge "recalling memory" when memory is actually populated (the NO-MEMORY
            # attempt runs recall mode too, but has nothing to retrieve -> no badge)
            has_mem = int(model.rag.current_size.item()) > 0
            badge = ("recalling memory" if (mode == "recall" and has_mem) else
                     ("write memory" if wrote else None))
            frames_out.append(_annotate_titled(obs["image"], title, tcol, badge))
        if wp < len(waypoints) - 1 and np.linalg.norm(obs["position"] - waypoints[wp]) < REACH:
            wp += 1
        if (done if complex_mode else (done or info["distance"] < REACH)):
            if frames_out is not None:                          # hold the solved frame a beat
                frames_out += [frames_out[-1]] * 6
            return 1
    return 0


def gate_rag_selflearning(model, W, device, decode_op, graph, featurize, m1_thr,
                          n_episodes, complex_mode=False, readout=None):
    """H7 lifelong learning: surprise-gated WRITE (experience) then RETRIEVE (recall) on
    the SAME layouts must improve success with NO weight update; a disjoint nominal set
    checks for interference."""
    def cfgs(base, n):
        if complex_mode:
            return [(0, 3, base + i) for i in range(n)]
        return [(i % 2, (i % 2) if (i // 2) % 2 == 0 else 1 - (i % 2), base + i) for i in range(n)]
    model.rag.current_size.zero_()                             # clear episodic memory
    learn = cfgs(5000, n_episodes); ctrl = cfgs(6000, n_episodes)
    rag = lambda seed, sr, gr, mode: run_episode_rag(
        model, W, seed, sr, gr, device, decode_op, graph, featurize, m1_thr, mode,
        complex_mode, readout=readout)
    ctrl_baseline = float(np.mean([rag(s, a, b, "recall") for a, b, s in ctrl]))  # empty memory
    exp = float(np.mean([rag(s, a, b, "experience") for a, b, s in learn]))       # writes on surprise
    mem = int(model.rag.current_size.item())
    rec = float(np.mean([rag(s, a, b, "recall") for a, b, s in learn]))           # memory used
    ctrl_after = float(np.mean([rag(s, a, b, "recall") for a, b, s in ctrl]))     # interference check
    return {"experience_success": exp, "recall_success": rec,
            "learning_gain": rec - exp, "memory_entries": mem,
            "control_before": ctrl_baseline, "control_after": ctrl_after,
            "interference": ctrl_baseline - ctrl_after,
            "passed": bool(rec - exp >= 0.05 and (ctrl_baseline - ctrl_after) <= 0.02)}


def gate_rag_lifelong_batches(model, W, device, decode_op, graph, featurize, m1_thr,
                               n_episodes, n_batches=5, complex_mode=False, readout=None):
    """H7 LIFELONG variant: iterate episode batches; after each batch's experience pass,
    recall improves on perturbed layouts; nominal layouts stay flat (no interference).

    Seeds: perturbed batches use ranges 7000+i*200 .. 7000+(i+1)*200  (i=0..n_batches-1)
           nominal control uses range 8000..8000+n_episodes (fixed, evaluated every batch)
    No weight updates at any point — purely episodic memory accumulation.
    Gate: perturbed recall rises ≥+0.10 from batch 1→3; nominal drop ≤ 0.02.
    """
    def cfgs(base, n):
        if complex_mode:
            return [(0, 3, base + i) for i in range(n)]
        return [(i % 2, 1 - (i % 2), base + i) for i in range(n)]
    rag = lambda seed, sr, gr, mode: run_episode_rag(
        model, W, seed, sr, gr, device, decode_op, graph, featurize, m1_thr, mode,
        complex_mode, readout=readout)

    model.rag.current_size.zero_()   # clear episodic memory before the lifelong run

    # Nominal (undisturbed) baseline — evaluate BEFORE any writes
    nom_cfgs = cfgs(8000, n_episodes)
    nominal_pre = float(np.mean([rag(s, a, b, "recall") for a, b, s in nom_cfgs]))

    batch_results = []
    for bi in range(n_batches):
        b_cfgs = cfgs(7000 + bi * n_episodes, n_episodes)
        before = float(np.mean([rag(s, a, b, "recall") for a, b, s in b_cfgs]))  # recall before writing
        float(np.mean([rag(s, a, b, "experience") for a, b, s in b_cfgs]))        # write on surprise
        after  = float(np.mean([rag(s, a, b, "recall") for a, b, s in b_cfgs]))  # recall after writing
        nom    = float(np.mean([rag(s, a, b, "recall") for a, b, s in nom_cfgs]))
        mem    = int(model.rag.current_size.item())
        batch_results.append({"batch": bi, "before": before, "after": after,
                               "gain": after - before, "nominal": nom, "memory": mem})
        print(f"  [H7 batch {bi}] perturbed before {before:.2f} -> after {after:.2f} "
              f"(gain {after-before:+.2f}) | nominal {nom:.2f} | memory {mem}")

    recalls = [r["after"] for r in batch_results]
    noms    = [r["nominal"] for r in batch_results]
    long_gain = recalls[-1] - recalls[0] if len(recalls) >= 2 else 0.0
    worst_interference = max((nominal_pre - n) for n in noms) if noms else 0.0
    passed = long_gain >= 0.10 and worst_interference <= 0.02
    return {"batches": batch_results, "nominal_pre": nominal_pre,
            "long_gain_first_to_last": long_gain,
            "worst_nominal_interference": worst_interference, "passed": passed}


def first_persistent_alarm(alarms, patience):
    run = 0
    for i, a in enumerate(alarms):
        run = run + 1 if a else 0
        if run >= patience:
            return i
    return None


# ---------------- protocol ----------------
def run(args):
    device = torch.device(args.device)
    model, W = load_model(args.model_path, device)
    frames, actions, positions, room_ids, starts = load_raw(args.data_path)
    total = frames.shape[0]
    # CRITICAL (instrument doctrine): every control step here encodes ONE frame; without BN
    # running-stat calibration the encoder is batch-dependent and single-frame latents are off
    # the probe's distribution (the bug that zeroed control project-wide). See encoders.py.
    from alps.evaluation.validate_temporal import calibrate_bn
    calibrate_bn(model, frames, device)
    print("[calibrate_bn] encoder BN running stats populated (single-frame inference fixed)")
    # Block-Rooms env variants must match training (see environment.py block_mode/wall/gate).
    global _ENV_KW
    _ENV_KW = dict(block_mode=getattr(args, "block_mode", False) or getattr(args, "block_wall", False)
                   or getattr(args, "block_gate", False),
                   block_wall=getattr(args, "block_wall", False),
                   block_gate=getattr(args, "block_gate", False),
                   block_radius=getattr(args, "block_radius", None),
                   block_step_scale=getattr(args, "block_step_scale", None))
    rng = np.random.RandomState(1)
    idx = rng.permutation(total)[: args.limit_samples] if args.limit_samples else rng.permutation(total)
    ntr = int(len(idx) * 0.8); tr = idx[:ntr]
    Ztr, _, _, _, Ptr = gather(model, frames, positions, room_ids, tr, device)
    decode_op = fit_probe(Ztr, Ptr, device)

    key_w, key_scale, hk = None, 6.0, None
    if args.complex:
        # H4: prefer label-free key detection (--label-free-key flag, or when no labels exist)
        hk_raw = load_has_keys(args.data_path)  # None if absent (unlabeled dataset)
        if getattr(args, "label_free_key", False) or hk_raw is None:
            print("[H4] using label-free key detector (surprise + VQ-flip) — no has_key labels used in planner")
            hk, surp_arr, vq_fl_arr = detect_key_pickups_unsup(
                model, frames, actions, starts, total, device,
                S=max(1, args.stride // 2))
            if hk_raw is not None:
                # Report H4 gate (precision/recall vs ground truth — measurement only)
                g_h4 = gate_h4_key_detector(hk, hk_raw, starts, total, S=max(1, args.stride // 2))
                print(f"[H4] label-free detector: prec {g_h4['precision']:.2f}  "
                      f"rec {g_h4['recall']:.2f}  f1 {g_h4['f1']:.2f}  "
                      f"-> {'PASS' if g_h4['passed'] else 'FAIL'}")
        else:
            hk = hk_raw
        if hk is not None:
            key_w = fit_key_probe(Ztr, hk[torch.from_numpy(tr)].float(), device)
        graph = build_graph_semantic(model, decode_op, key_w, key_scale, frames, positions,
                                     room_ids, starts, total, device, k=args.fine_k,
                                     S=max(1, args.stride // 2), has_keys=hk)
        featurize = make_featurize(decode_op, key_w, key_scale, device)
    else:
        graph = build_graph_raw(model, decode_op, frames, positions, room_ids, starts,
                                total, device, k=args.fine_k, S=args.stride)
        featurize = lambda z: z
    cz = graph.z_centroids if graph.z_centroids is not None else graph.centroids
    ZC = torch.tensor(cz, device=device)

    # SPATIAL: rebuild graph/decode/ZC on the position-faithful spatial readout. Under pure
    # SSL the global pool is position-blind (it discards the small agent), so monitors and
    # predictor-decoded control run on the coarse gxg spatial readout instead.
    readout, decode_state = None, None
    if getattr(args, "spatial", False):
        g = args.spatial_grid
        readout = lambda z: model.spatial_readout(z, grid=g)

        @torch.no_grad()
        def _gr(ix):
            o = []
            for c0 in range(0, len(ix), 128):
                b = torch.as_tensor(np.asarray(ix[c0:c0 + 128]))
                o.append(readout(model.encode_frame(frames[b].to(device).float() / 255.0)).cpu())
            return torch.cat(o)

        ridge = fit_ridge_decode(_gr(tr), Ptr, device)
        decode_state = lambda grid: ridge(model.spatial_readout(grid, grid=g))
        graph = build_graph_raw(model, ridge, frames, positions, room_ids, starts, total,
                                device, k=args.fine_k, S=max(1, args.stride // 2), readout=readout)
        graph.decoded_xy = ridge(torch.tensor(graph.centroids, device=device)).cpu().numpy()
        ZC = torch.tensor(graph.centroids, device=device)
        decode_op = decode_state    # RAG control reads decode_op on the spatial readout
        print(f"--- [SPATIAL {g}x{g}] fourth-brain monitors + control on the spatial readout ---")

    def cfgs(base, n):
        if args.complex:
            return [(0, 3, base + i) for i in range(n)]
        return [(i % 2, (i % 2) if (i // 2) % 2 == 0 else 1 - (i % 2), base + i) for i in range(n)]

    ep = lambda seed, sr, gr, policy, thr, fb: run_episode_fb(
        model, W, seed, sr, gr, device, graph, ZC, policy,
        thr=thr, complex_mode=args.complex, alarm_k=args.alarm_k,
        patience=args.patience, grace=args.grace, stall_w=args.stall_w, fallback=fb,
        readout=readout, decode_state=decode_state)

    out = {}
    # PASS A — calibration (fixed tactical policy, monitors only)
    cal = [ep(seed, sr, gr, "tactical", None, "record") for sr, gr, seed in cfgs(3000, args.n_cal)]
    good = [r for r in cal if r["success"]] or cal

    def _q(vals, q, default):
        v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
        return float(np.quantile(v, q)) if v.size else float(default)

    th = {"m1": _q(np.concatenate([r["m1"] for r in good]) if good else [], 0.90, 1e9),
          "m2": _q(np.concatenate([r["m2"] for r in good]) if good else [], 0.90, 1e9),
          "m3": _q([v for r in good for v in r["m3"]], 0.10, 0.0)}
    out["thresholds"] = {**th, "cal_success_rate": float(np.mean([r["success"] for r in cal]))}

    # ── --video: render the SELF-MONITOR -> ESCALATION -> FALLBACK story as gifs (H8-H10) ──
    # (--rag/--h7-lifelong route --video to their own before/after renderers below instead)
    if getattr(args, "video", False) and not getattr(args, "rag", False) \
            and not getattr(args, "h7_lifelong", False):
        from PIL import Image
        vdir = os.path.join(args.save_dir, "videos_fourthbrain")
        os.makedirs(vdir, exist_ok=True)
        print(f"--- [video] rendering monitored episodes (escalation+fallback) to {vdir} ---")
        n_saved = 0
        for sr, gr, seed in cfgs(6000, max(40, args.n_eval * 3)):
            if n_saved >= getattr(args, "n_video", 4):
                break
            fo = []
            r = run_episode_fb(model, W, seed, sr, gr, device, graph, ZC, "escalation",
                               thr=th, complex_mode=args.complex, alarm_k=args.alarm_k,
                               patience=args.patience, grace=args.grace, stall_w=args.stall_w,
                               fallback="on", readout=readout, decode_state=decode_state,
                               frames_out=fo)
            escalated = max(r["tiers_used"]) >= 1        # tier moved past bare operative
            if fo and (getattr(args, "video_all", False) or escalated):
                gif = os.path.join(vdir, f"fb_{'complex' if args.complex else 'block'}_seed{seed}.gif")
                ims = [Image.fromarray(f) for f in fo]
                ims[0].save(gif, save_all=True, append_images=ims[1:],
                            duration=int(1000 / getattr(args, "fps", 8)), loop=0)
                print(f"  [video seed{seed}] tiers={r['tiers_used']} fb_exec={r['fb_executed']} "
                      f"safe_reached={r['safe_reached']} -> {gif}")
                n_saved += 1
        if n_saved == 0:
            print("  [video] no escalation episodes found (model rarely triggers); "
                  "use --video-all to film all attempts.")
        return out

    # ── H7 RAG lifelong (--h7-lifelong): learning curve across batches ──
    if getattr(args, "h7_lifelong", False):
        print("--- [H7 LIFELONG] RAG self-learning across episode batches ---")
        n_batches = getattr(args, "n_batches", 5)
        rg = gate_rag_lifelong_batches(model, W, device, decode_op, graph, featurize,
                                        th["m1"], args.n_eval, n_batches=n_batches,
                                        complex_mode=args.complex, readout=readout)
        out["H7_lifelong"] = rg
        os.makedirs(args.save_dir, exist_ok=True)
        p = os.path.join(args.save_dir, "rag_lifelong_complex.json" if args.complex
                         else "rag_lifelong.json")
        with open(p, "w") as f:
            json.dump(out, f, indent=2, default=float)
        print(f"H7 lifelong: gain {rg['long_gain_first_to_last']:+.2f} "
              f"interference {rg['worst_nominal_interference']:+.2f} "
              f"-> {'PASS' if rg['passed'] else 'FAIL'}")
        print(f"[report] {p}")
        return out

    # ── H7 RAG single-pass (--rag): original experience/recall/control ──
    if getattr(args, "rag", False):
        if getattr(args, "video", False):
            # SELF-LEARNING video: same layout BEFORE (empty memory) vs AFTER (memory written by
            # surprise-gated experience) -> the agent improves with NO weight update.
            vdir = os.path.join(args.save_dir, "videos_rag"); os.makedirs(vdir, exist_ok=True)
            print(f"--- [rag-video] before/after self-learning clips -> {vdir} ---")
            def _cfgv(base, n):
                if args.complex:
                    return [(0, 3, base + i) for i in range(n)]
                return [(i % 2, (i % 2) if (i // 2) % 2 == 0 else 1 - (i % 2), base + i) for i in range(n)]
            learn_v = _cfgv(5000, max(12, args.n_eval))
            model.rag.current_size.zero_()                          # empty memory
            befores = []
            for sr_, gr_, seed_ in learn_v:                         # attempt 1: no memory
                fb_ = []
                ok_b = run_episode_rag(model, W, seed_, sr_, gr_, device, decode_op, graph,
                                       featurize, th["m1"], "recall", args.complex, readout=readout,
                                       frames_out=fb_, title="NO MEMORY (attempt 1)", tcol=(120, 55, 55))
                befores.append((seed_, sr_, gr_, fb_, ok_b))
            for sr_, gr_, seed_ in learn_v:                         # write memory on surprise
                run_episode_rag(model, W, seed_, sr_, gr_, device, decode_op, graph, featurize,
                                th["m1"], "experience", args.complex, readout=readout)
            saved = 0
            for seed_, sr_, gr_, fb_, ok_b in befores:              # attempt 2: memory recalled
                if saved >= getattr(args, "n_video", 4):
                    break
                fa_ = []
                ok_a = run_episode_rag(model, W, seed_, sr_, gr_, device, decode_op, graph,
                                       featurize, th["m1"], "recall", args.complex, readout=readout,
                                       frames_out=fa_, title="AFTER RAG RECALL (attempt 2)", tcol=(50, 120, 70))
                if fb_ and fa_ and (getattr(args, "video_all", False) or (ok_a and not ok_b)):
                    path = os.path.join(vdir, f"rag_{'complex' if args.complex else 'block'}_seed{seed_}.gif")
                    _save_sidebyside(fb_, fa_, path, fps=getattr(args, "fps", 8))
                    print(f"  [rag-video seed{seed_}] before {'SOLVED' if ok_b else 'stalled'} | "
                          f"after {'SOLVED' if ok_a else 'stalled'}"
                          f"{'  <<< LEARNED (no weight update)' if (ok_a and not ok_b) else ''} -> {path}")
                    saved += 1
            if saved == 0:
                print("  [rag-video] no before-fail/after-solve pairs found (use --video-all to film all).")
        rg = gate_rag_selflearning(model, W, device, decode_op, graph, featurize,
                                   th["m1"], args.n_eval, complex_mode=args.complex, readout=readout)
        out["H7_rag_selflearning"] = rg
        os.makedirs(args.save_dir, exist_ok=True)
        p = os.path.join(args.save_dir, "rag_selflearning_complex.json" if args.complex
                         else "rag_selflearning.json")
        with open(p, "w") as f:
            json.dump(out, f, indent=2, default=float)
        print("\n===== RAG-IN-THE-LOOP SELF-LEARNING (H7, surprise-gated) =====")
        print(f"  surprise thr m1>{th['m1']:.3f} | memory entries {rg['memory_entries']}")
        print(f"  experience {rg['experience_success']:.2f} -> recall {rg['recall_success']:.2f} "
              f"(gain {rg['learning_gain']:+.2f})")
        print(f"  control before {rg['control_before']:.2f} -> after {rg['control_after']:.2f} "
              f"(interference {rg['interference']:+.2f}) -> {'PASS' if rg['passed'] else 'FAIL'}")
        print(f"[report] {p}")
        return out

    # PASS B — H8 on held-out: do monitors predict failure?
    ho = [ep(seed, sr, gr, "tactical", th, "record") for sr, gr, seed in cfgs(4000, args.n_eval)]
    fail = [0 if r["success"] else 1 for r in ho]
    frac_alarm = [float(np.mean(r["alarms"])) for r in ho]
    out["H8_monitoring"] = {
        "auroc_frac_alarm": auroc(frac_alarm, fail),
        "auroc_mean_m1": auroc([float(np.mean(r["m1"])) for r in ho], fail),
        "auroc_mean_m2": auroc([float(np.mean(r["m2"])) for r in ho], fail),
        "auroc_frac_stall": auroc([float(np.mean([v < th["m3"] for v in r["m3"] if np.isfinite(v)] or [0])) for r in ho], fail),
        "median_lead_steps": float(np.median([r["steps"] - fa for r in ho if not r["success"]
                                              and (fa := first_persistent_alarm(r["alarms"], args.patience)) is not None] or [0])),
        "n_eval": len(ho), "failure_rate": float(np.mean(fail)),
    }
    h8 = out["H8_monitoring"]
    h8["auroc_best"] = float(np.nanmax([h8["auroc_frac_alarm"], h8["auroc_mean_m1"],
                                        h8["auroc_mean_m2"], h8["auroc_frac_stall"]]))
    # gate: at least one calibrated monitor predicts failure with early lead
    h8["passed"] = bool(h8["auroc_best"] >= 0.8 and h8["median_lead_steps"] >= 10)

    # PASS C — H9 escalation vs fixed tiers (+ false-trigger rate)
    rop = [ep(seed, sr, gr, "operative", th, "record") for sr, gr, seed in cfgs(4000, args.n_eval)]
    res = [ep(seed, sr, gr, "escalation", th, "record") for sr, gr, seed in cfgs(4000, args.n_eval)]
    sr_ = lambda rs: float(np.mean([r["success"] for r in rs]))
    succ_would = [r for r in res if r["success"] and r["fb_would_step"] is not None]
    out["H9_escalation"] = {
        "success_operative": sr_(rop), "success_tactical": sr_(ho), "success_escalation": sr_(res),
        "edge_over_best_fixed": sr_(res) - max(sr_(rop), sr_(ho)),
        "false_trigger_rate": len(succ_would) / max(1, sum(r["success"] for r in res)),
        "tier_usage": {str(t): float(np.mean([t in r["tiers_used"] for r in res])) for t in range(4)},
    }
    out["H9_escalation"]["passed"] = bool(out["H9_escalation"]["edge_over_best_fixed"] >= 0.05
                                          and out["H9_escalation"]["false_trigger_rate"] <= 0.10)

    # PASS D — H10 fallback safe state vs random-walk baseline
    rfb = [ep(seed, sr, gr, "escalation", th, "on") for sr, gr, seed in cfgs(4000, args.n_eval)]
    rrw = [ep(seed, sr, gr, "escalation", th, "random") for sr, gr, seed in cfgs(4000, args.n_eval)]
    trig = [r for r in rfb if r["fb_executed"]]
    trig_rw = [r for r in rrw if r["fb_executed"]]
    out["H10_fallback"] = {
        "trigger_rate": len(trig) / len(rfb),
        "safe_reach_when_triggered": float(np.mean([r["safe_reached"] for r in trig])) if trig else None,
        "safe_reach_random_baseline": float(np.mean([r["safe_reached"] for r in trig_rw])) if trig_rw else None,
        "mean_dist_safe_end_fb": float(np.mean([r["dist_safe_end"] for r in trig])) if trig else None,
        "mean_dist_safe_end_rw": float(np.mean([r["dist_safe_end"] for r in trig_rw])) if trig_rw else None,
        "n_triggered": len(trig),
    }
    p_ = out["H10_fallback"]
    out["H10_fallback"]["passed"] = bool(trig and p_["safe_reach_when_triggered"] is not None
                                         and p_["safe_reach_when_triggered"] >= 0.8
                                         and (p_["safe_reach_random_baseline"] is None
                                              or p_["safe_reach_when_triggered"] > p_["safe_reach_random_baseline"]))

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "fourth_brain_complex.json" if args.complex else "fourth_brain.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)

    h8, h9, h10 = out["H8_monitoring"], out["H9_escalation"], out["H10_fallback"]
    print("\n===== FOURTH BRAIN (monitor -> escalate -> fallback) =====")
    print(f"thr  m1>{th['m1']:.3f} m2>{th['m2']:.3f} stall<{th['m3']:.3f} | cal success {out['thresholds']['cal_success_rate']:.2f}")
    print(f"H8   AUROC best {h8['auroc_best']:.2f} (alarm {h8['auroc_frac_alarm']:.2f} m1 {h8['auroc_mean_m1']:.2f} "
          f"m2 {h8['auroc_mean_m2']:.2f} stall {h8['auroc_frac_stall']:.2f}) | lead {h8['median_lead_steps']:.0f} steps "
          f"-> {'PASS' if h8['passed'] else 'FAIL'}")
    print(f"H9   success op {h9['success_operative']:.2f} | tac {h9['success_tactical']:.2f} | "
          f"ESCALATION {h9['success_escalation']:.2f} (edge {h9['edge_over_best_fixed']:+.2f}) | "
          f"false-trigger {h9['false_trigger_rate']:.2f} -> {'PASS' if h9['passed'] else 'FAIL'}")
    print(f"H10  trigger {h10['trigger_rate']:.2f} | safe-reach FB {h10['safe_reach_when_triggered']} "
          f"vs random {h10['safe_reach_random_baseline']} -> {'PASS' if h10['passed'] else 'FAIL'}")
    print(f"[report] {path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="results/two_rooms/validation/temporal_world_model.pt")
    ap.add_argument("--data-path", default="data/two_rooms/trajectories.pt")
    ap.add_argument("--save-dir", default="results/two_rooms/validation")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-cal", type=int, default=24)
    ap.add_argument("--n-eval", type=int, default=24)
    ap.add_argument("--limit-samples", type=int, default=4000)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--fine-k", type=int, default=24)
    ap.add_argument("--alarm-k", type=int, default=2)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--grace", type=int, default=6)
    ap.add_argument("--stall-w", type=int, default=8)
    ap.add_argument("--complex", action="store_true")
    ap.add_argument("--spatial", action="store_true",
                    help="Run monitors + control on the coarse gxg SPATIAL readout (the "
                         "global pool is position-blind under pure SSL).")
    ap.add_argument("--spatial-grid", type=int, default=8, help="spatial readout grid size (g x g)")
    ap.add_argument("--label-free-key", action="store_true",
                    help="H4: use the label-free key detector (surprise + VQ-flip) instead "
                         "of dataset has_key labels to build the key landmark in the graph. "
                         "Closes the planner-side supervision leak for the unsupervised claim.")
    ap.add_argument("--rag", action="store_true",
                    help="H7 single-pass: run the RAG-in-the-loop self-learning gate "
                         "(surprise-gated memory write/retrieve) instead of H8-H10")
    ap.add_argument("--h7-lifelong", action="store_true",
                    help="H7 LIFELONG: run RAG across multiple episode batches and report "
                         "the learning curve (gain by batch) and nominal interference")
    ap.add_argument("--n-batches", type=int, default=5,
                    help="number of batches for --h7-lifelong (default 5)")
    ap.add_argument("--block-mode", action="store_true", help="Block-Rooms env (match training)")
    ap.add_argument("--block-wall", action="store_true", help="Block-Rooms WALL+GAP (match training)")
    ap.add_argument("--block-gate", action="store_true", help="Block-Rooms SWITCH-GATE (match training)")
    ap.add_argument("--block-radius", type=float, default=None, help="block render radius (match training)")
    ap.add_argument("--block-step-scale", type=float, default=None, help="block step scale (match training)")
    ap.add_argument("--video", action="store_true",
                    help="render the self-monitor -> escalation -> fallback story as gifs "
                         "(H8-H10 made visible: monitors spike -> tier escalates -> fallback).")
    ap.add_argument("--video-all", action="store_true",
                    help="film every episode, not just ones where the monitor escalated")
    ap.add_argument("--n-video", type=int, default=4, help="number of monitored gifs to save")
    ap.add_argument("--fps", type=int, default=8)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
