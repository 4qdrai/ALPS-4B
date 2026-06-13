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
    load_model, gather, HistoryBuffer, hist_greedy_action, build_graph_raw,
    build_graph_semantic, fit_key_probe, make_featurize, load_has_keys, REACH)

TIERS = ("operative", "tactical", "strategic", "fallback")


# ---------------- helpers ----------------
@torch.no_grad()
def predicted_pooled(model, buf, a, device):
    """Pooled latent the operative predictor expects after action `a`."""
    z_hist = torch.stack(buf.z, dim=1)                                   # [1,W,N,D]
    a_idx = buf.a[1:] + [a]
    a_hist = F.one_hot(torch.tensor(a_idx, device=device), 4).float().unsqueeze(0)
    return model.pool(model.op_predict_next(z_hist, a_hist)).squeeze(0)  # [D]


@torch.no_grad()
def plan_waypoints(model, graph, featurize, buf, goal_xy, seed, gr, complex_mode,
                   has_key, device):
    """Fine-graph waypoint plan from the CURRENT state (mirrors run_episode_4b's
    planning block incl. the mandatory key waypoint when the key is still needed)."""
    eg = TwoRoomsEnv(seed=seed, complex_mode=complex_mode, hazards=False)
    eg.reset() if complex_mode else eg.reset(start_room=gr, goal_room=gr)
    eg.agent_pos = goal_xy.copy()
    if complex_mode:
        eg.has_key = True
    zg = model.pool(model.encode_frame(obs_to_frame({"image": eg.render()}, device).unsqueeze(0))).squeeze(0).cpu().numpy()
    zs = model.pool(buf.cur_z).squeeze(0).cpu().numpy()
    sn, gn = graph.node_of_latent(featurize(zs)), graph.node_of_latent(featurize(zg))
    if complex_mode and graph.key_node is not None and not has_key:
        p1 = graph.shortest_path(sn, graph.key_node) or [graph.key_node]
        p2 = graph.shortest_path(graph.key_node, gn) or [gn]
        path = p1 + p2[1:]
    else:
        path = graph.shortest_path(sn, gn) or [gn]
    seg = path[1:] if len(path) > 1 else path
    return [graph.decoded_xy[n] for n in seg] + [goal_xy.copy()]


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
def run_episode_fb(model, W, seed, sr, gr, device, decode_op, graph, featurize, ZC,
                   policy, thr=None, complex_mode=False, max_steps=140,
                   alarm_k=2, patience=4, grace=6, stall_w=8, fallback="record"):
    """policy in {operative, tactical, escalation}. Monitors always logged.
    fallback: 'record' = note the would-trigger step but keep trying (tier<=2);
              'on'     = execute the safe-state routing at tier 3;
              'random' = random actions after the trigger (H10 baseline)."""
    env = TwoRoomsEnv(seed=seed, complex_mode=complex_mode, hazards=False)
    obs = env.reset() if complex_mode else env.reset(start_room=sr, goal_room=gr)
    goal_xy = obs["target"].copy()
    buf = HistoryBuffer(model, W, device); buf.reset(obs_to_frame(obs, device))
    safe = safe_node_of(graph); safe_xy = graph.decoded_xy[safe]

    tier = 0 if policy in ("operative", "escalation") else 1
    waypoints, wp = None, 0
    if tier == 1:
        waypoints = plan_waypoints(model, graph, featurize, buf, goal_xy, seed, gr,
                                   complex_mode, bool(obs.get("has_key", 0)), device)

    m1s, m2s, m3s, alarms, tiers_used = [], [], [], [], {0}
    consec, since_esc = 0, grace
    zp_prev = None
    pos_hist = [obs["position"].copy()]
    fb_would, fb_active, fb_wp, fb_idx = None, False, None, 0

    for s in range(max_steps):
        # ---- choose sub-goal by current tier ----
        if fb_active and fallback == "on":
            sub = fb_wp[min(fb_idx, len(fb_wp) - 1)]
        elif tier == 0:
            sub = goal_xy
        else:  # tier 1/2 follow the (re)planned waypoints
            sub = waypoints[min(wp, len(waypoints) - 1)]
        if fb_active and fallback == "random":
            a = int(np.random.RandomState(seed * 7919 + s).randint(0, 4))
        else:
            a = hist_greedy_action(buf, decode_op, sub, device)

        # ---- monitor m1 needs the prediction made BEFORE stepping ----
        zp = predicted_pooled(model, buf, a, device)
        obs, _, done, info = env.step(a); buf.push(obs_to_frame(obs, device), a)
        za = model.pool(buf.cur_z).squeeze(0)
        m1 = float((zp - za).norm()) if zp_prev is not None or True else 0.0
        m2 = float(torch.cdist(za.unsqueeze(0), ZC).min())
        pos_hist.append(obs["position"].copy())
        m3 = float(np.linalg.norm(pos_hist[-1] - pos_hist[-stall_w])) if len(pos_hist) > stall_w else float("inf")
        m1s.append(m1); m2s.append(m2); m3s.append(m3)
        zp_prev = zp

        alarmed = False
        if thr is not None:
            score = int(m1 > thr["m1"]) + int(m2 > thr["m2"]) + int(m3 < thr["m3"])
            alarmed = score >= alarm_k
        alarms.append(bool(alarmed))
        consec = consec + 1 if alarmed else 0
        since_esc += 1

        # ---- waypoint advance ----
        if fb_active and fallback == "on":
            if fb_idx < len(fb_wp) - 1 and np.linalg.norm(obs["position"] - fb_wp[fb_idx]) < REACH:
                fb_idx += 1
            if np.linalg.norm(obs["position"] - safe_xy) < 1.0:
                return _result(False, s + 1, sr, gr, complex_mode, m1s, m2s, m3s, alarms,
                               fb_would, True, True, float(np.linalg.norm(obs["position"] - safe_xy)),
                               tiers_used)
        elif waypoints is not None and wp < len(waypoints) - 1 and \
                np.linalg.norm(obs["position"] - waypoints[wp]) < REACH:
            wp += 1

        # ---- escalation ----
        if policy == "escalation" and thr is not None and consec >= patience and since_esc >= grace \
                and not fb_active:
            consec, since_esc = 0, 0
            if tier < 2:
                tier += 1; tiers_used.add(tier)
                waypoints = plan_waypoints(model, graph, featurize, buf, goal_xy, seed, gr,
                                           complex_mode, bool(obs.get("has_key", 0)), device)
                wp = 0
            else:
                if fb_would is None:
                    fb_would = s
                if fallback in ("on", "random"):
                    fb_active = True; tiers_used.add(3)
                    if fallback == "on":
                        zs = model.pool(buf.cur_z).squeeze(0).cpu().numpy()
                        pth = graph.shortest_path(graph.node_of_latent(featurize(zs)), safe) or [safe]
                        seg = pth[1:] if len(pth) > 1 else pth
                        fb_wp, fb_idx = [graph.decoded_xy[n] for n in seg], 0

        reached = done if complex_mode else (done or info["distance"] < REACH)
        if reached and not fb_active:
            return _result(True, s + 1, sr, gr, complex_mode, m1s, m2s, m3s, alarms,
                           fb_would, False, False, float(np.linalg.norm(obs["position"] - safe_xy)),
                           tiers_used)
    return _result(False, max_steps, sr, gr, complex_mode, m1s, m2s, m3s, alarms,
                   fb_would, fb_active, bool(np.linalg.norm(obs["position"] - safe_xy) < 1.0),
                   float(np.linalg.norm(obs["position"] - safe_xy)), tiers_used)


def _result(success, steps, sr, gr, cx, m1s, m2s, m3s, alarms, fb_would, fb_active,
            safe_reached, dist_safe, tiers_used):
    return {"success": success, "steps": steps, "is_cross": True if cx else sr != gr,
            "m1": m1s, "m2": m2s, "m3": m3s, "alarms": alarms,
            "fb_would_step": fb_would, "fb_executed": fb_active,
            "safe_reached": safe_reached, "dist_safe_end": dist_safe,
            "tiers_used": sorted(tiers_used)}


@torch.no_grad()
def run_episode_rag(model, W, seed, sr, gr, device, decode_op, graph, featurize, m1_thr,
                    mode, complex_mode=False, max_steps=140):
    """Latent-RAG in the control loop, gated by the SELF-MONITOR surprise signal m1.
      mode 'experience' : act; when surprise m1 > thr (the monitor fires), WRITE memory
                          key=context-latent, value=correction (actual_next - predicted).
      mode 'recall'     : act, but first RETRIEVE a correction for the current context
                          and APPLY it to the operative prediction (surprise-gated by the
                          RAG similarity threshold) -> memory steers control.
    Returns success (1/0)."""
    env = TwoRoomsEnv(seed=seed, complex_mode=complex_mode, hazards=False)
    obs = env.reset() if complex_mode else env.reset(start_room=sr, goal_room=gr)
    goal_xy = obs["target"].copy()
    buf = HistoryBuffer(model, W, device); buf.reset(obs_to_frame(obs, device))
    waypoints = plan_waypoints(model, graph, featurize, buf, goal_xy, seed, gr,
                               complex_mode, bool(obs.get("has_key", 0)), device)
    wp = 0
    for s in range(max_steps):
        sub = torch.tensor(waypoints[min(wp, len(waypoints) - 1)], device=device, dtype=torch.float32)
        ctx = buf.cur_z                                         # context BEFORE the step
        z_hist = torch.stack(buf.z, dim=1)
        best_a, best_d, best_pred = 0, 1e30, None
        for a in range(4):
            a_idx = buf.a[1:] + [a]
            a_hist = F.one_hot(torch.tensor(a_idx, device=device), 4).float().unsqueeze(0)
            z_next = model.op_predict_next(z_hist, a_hist)
            if mode == "recall":
                z_next = model.rag_correct(z_next, ctx)         # surprise-gated by RAG sim
            d = float((decode_op(z_next)[0] - sub).norm())
            if d < best_d:
                best_d, best_a, best_pred = d, a, model.pool(z_next).squeeze(0)
        obs, _, done, info = env.step(best_a); buf.push(obs_to_frame(obs, device), best_a)
        za = model.pool(buf.cur_z).squeeze(0)
        m1 = float((best_pred - za).norm())
        if mode == "experience" and m1 > m1_thr:               # monitor fires -> memorize
            model.rag.write_memory(model.pool(ctx).squeeze(0), za - best_pred)
        if wp < len(waypoints) - 1 and np.linalg.norm(obs["position"] - waypoints[wp]) < REACH:
            wp += 1
        if (done if complex_mode else (done or info["distance"] < REACH)):
            return 1
    return 0


def gate_rag_selflearning(model, W, device, decode_op, graph, featurize, m1_thr,
                          n_episodes, complex_mode=False):
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
        model, W, seed, sr, gr, device, decode_op, graph, featurize, m1_thr, mode, complex_mode)
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
    rng = np.random.RandomState(1)
    idx = rng.permutation(total)[: args.limit_samples] if args.limit_samples else rng.permutation(total)
    ntr = int(len(idx) * 0.8); tr = idx[:ntr]
    Ztr, _, _, _, Ptr = gather(model, frames, positions, room_ids, tr, device)
    decode_op = fit_probe(Ztr, Ptr, device)

    key_w, key_scale, hk = None, 6.0, None
    if args.complex:
        hk = load_has_keys(args.data_path)
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

    def cfgs(base, n):
        if args.complex:
            return [(0, 3, base + i) for i in range(n)]
        return [(i % 2, (i % 2) if (i // 2) % 2 == 0 else 1 - (i % 2), base + i) for i in range(n)]

    ep = lambda seed, sr, gr, policy, thr, fb: run_episode_fb(
        model, W, seed, sr, gr, device, decode_op, graph, featurize, ZC, policy,
        thr=thr, complex_mode=args.complex, alarm_k=args.alarm_k,
        patience=args.patience, grace=args.grace, stall_w=args.stall_w, fallback=fb)

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

    # ── RAG-in-the-loop self-learning (surprise-gated memory + monitoring) ──
    if getattr(args, "rag", False):
        rg = gate_rag_selflearning(model, W, device, decode_op, graph, featurize,
                                   th["m1"], args.n_eval, complex_mode=args.complex)
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
    ap.add_argument("--rag", action="store_true",
                    help="run the RAG-in-the-loop self-learning gate (H7): surprise-gated "
                         "memory write/retrieve in the control loop, instead of H8-H10")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
