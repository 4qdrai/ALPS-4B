"""diagnose_control.py -- WHY does predictor-decoded control fail even with a sharp decode?

The four-brain control selects an action by asking the trained op-predictor "where would each
action take me?" (op_predict_next -> spatial-readout -> frozen ridge -> predicted position) and
stepping toward the planned sub-goal. That only works if the predictor's action->position
mapping is (a) action-SENSITIVE and (b) ACCURATE. This probes both directly:

  G1_spatial    : real-frame spatial decode (reference; should be sharp, ~0.2-0.35 big-agent).
  action_spread : std of the predicted next position ACROSS the 4 actions (wu). ~0 => the
                  predictor IGNORES the action (control can't discriminate). True per-step motion
                  ~0.27wu; over the predictor's stride-step ~1wu, so a healthy spread is ~0.5-1wu.
  pred_err      : || predicted next pos - the env's TRUE next pos || per action (wu). Predictor
                  forward accuracy in world units.
  direction_acc : fraction of steps where the predictor's best action toward the goal equals the
                  env's TRUE best action. 1.0 perfect, 0.25 random. <~0.5 => the predictor cannot
                  drive greedy control (the smoking gun for operative=0).

Usage:
  PYTHONPATH=src python -m alps.evaluation.diagnose_control \
      --model-path results/two_rooms/validation/unsup_temporal.pt \
      --data-path  data/two_rooms/trajectories_unsup.pt --spatial-grid 8 --n-steps 200
"""
import sys; sys.path.insert(0, "src")
import argparse, copy
import numpy as np
import torch
import torch.nn.functional as F

from alps.benchmarks.two_rooms.environment import TwoRoomsEnv
from alps.benchmarks.two_rooms.world_model_planning import obs_to_frame
from alps.training.train_hier import load_raw
from alps.evaluation.validate_temporal import (load_model, fit_ridge_decode, HistoryBuffer, calibrate_bn,
                                               fit_softargmax_decode, fit_calibrated_softargmax,
                                               _gather_token_grids, gather_pred_grids)
from alps.core.slot_readout import fit_slot_decode, fit_eq_slot_probe


@torch.no_grad()
def fit_calibrated_decode(m, frames, positions, actions, starts, tot, readout, W, dev, n_win=3000):
    """Fit the ridge on the PREDICTOR's OWN outputs (op_predict_next -> spatial readout) paired
    with the TRUE next position -- not on real frames. The control decodes predicted latents,
    which are off the real-frame manifold; calibrating on predictions removes any systematic
    linear (sign/scale/offset) distortion. Still a frozen label-free probe (measuring instrument)."""
    E = starts.shape[0]; rng = np.random.RandomState(1); ss = []
    for _ in range(n_win):
        e = rng.randint(E - 1); s = int(starts[e]); end = int(starts[e + 1]) if e + 1 < E else tot
        if end - s >= W + 1:
            ss.append(rng.randint(s, end - W))
    # batch scales INVERSELY with token count: the predictor's block-causal attention over the
    # W-frame context is O((W*N)^2) memory; bs=64 fits patch-16 (N=64) but OOMs 8GB at patch-8
    # (N=256, ~16x). One encode probes N to set a safe batch (fixes the egocentric-patch8 OOM).
    ss = np.array(ss); Xs, Ys = [], []
    N0 = m.encode_frame(frames[torch.as_tensor(ss[:1])].to(dev).float() / 255.).shape[1]
    bs = max(4, int(64 * 64 / N0))
    for c in range(0, len(ss), bs):
        sb = ss[c:c + bs]
        fidx = np.stack([sb + k for k in range(W + 1)], 1)              # [B,W+1]
        fr = frames[torch.as_tensor(fidx.reshape(-1))].to(dev).float() / 255.
        z = m.encode_frame(fr); N, D = z.shape[1], z.shape[2]
        z = z.reshape(len(sb), W + 1, N, D)
        a_hist = F.one_hot(actions[torch.as_tensor(fidx[:, :W].reshape(-1))].to(dev).long(), 4).float().reshape(len(sb), W, 4)
        z_pred = m.op_predict_next(z[:, :W], a_hist)                    # [B,N,D]
        Xs.append(readout(z_pred).cpu()); Ys.append(positions[torch.as_tensor(fidx[:, W])])
    return fit_ridge_decode(torch.cat(Xs), torch.cat(Ys), dev)


@torch.no_grad()
def fit_forward_probe(m, frames, positions, actions, starts, tot, readout, ridge, dev, n=40000):
    """Frozen forward-dynamics model in the DECODED-STATE space (DINO-WM recipe, robust version):
    for each action a, ridge( decoded_pos(z_t) ) -> true position_{t+1}, from observed 1-step
    transitions. The state is the frozen unsupervised position read-out (2-D -> CANNOT overfit,
    unlike the 12k-D latent), the dynamics is learned per action from the agent's OWN transitions
    (proprioceptive, label-free). The encoder stays untouched/unsupervised; this is a frozen control
    instrument like the position read-out. Returns fn(decoded_pos[..,2], action:int) -> next pos."""
    E = starts.shape[0]; rng = np.random.RandomState(2)
    ti = {a: [] for a in range(4)}; tj = {a: [] for a in range(4)}
    while sum(len(v) for v in ti.values()) < n:
        e = rng.randint(E - 1); s = int(starts[e]); end = int(starts[e + 1]) if e + 1 < E else tot
        if end - s < 2:
            continue
        t = rng.randint(s, end - 1); a = int(actions[t].item())
        ti[a].append(t); tj[a].append(t + 1)
    dec = {}
    for a in range(4):
        if len(ti[a]) < 50:
            continue
        I = torch.as_tensor(np.asarray(ti[a])); J = torch.as_tensor(np.asarray(tj[a]))
        P = []
        for c in range(0, len(I), 128):
            b = I[c:c + 128]
            P.append(ridge(readout(m.encode_frame(frames[b].to(dev).float() / 255.))).cpu())  # decoded current pos
        dec[a] = fit_ridge_decode(torch.cat(P), positions[J], dev, lam=1.0)                    # 2-D -> 2-D
    return lambda dp, a: dec[a](dp)


class SlotHistoryBuffer:
    """Slot-mode control buffer that keeps slot computation IN THE TRAINED REGIME: the model
    only ever chains slots over W-frame windows (slots_of_window), so control must too. An
    unbounded episode-long chain drifts out of distribution (measured: probe G1 0.49 on
    window slots, but 2.8 in a loop running a 40-step chain). Stores the last W token grids
    and recomputes the W-frame slot chain each step; `peek_frame_slots` evaluates a
    counterfactual next frame in the same regime without mutating state."""
    def __init__(self, m, W, dev, readout):
        self.m, self.W, self.dev, self.readout = m, W, dev, readout
        self.zg, self.a = [], []

    def reset(self, frame):
        z = self.m.encode_frame(frame.unsqueeze(0))
        self.zg = [z] * self.W; self.a = [0] * self.W
        self._rebuild()

    def push(self, frame, action):
        z = self.m.encode_frame(frame.unsqueeze(0))
        self.zg = (self.zg + [z])[-self.W:]; self.a = (self.a + [action])[-self.W:]
        self._rebuild()

    def _rebuild(self):
        self.s = self.m.slots_of_window(torch.stack(self.zg, dim=1))    # [1,W,K,D]

    @property
    def cur_z(self):
        return self.s[:, -1]                                            # [1,K,D]

    def _a_hist(self, last_action):
        return F.one_hot(torch.tensor(self.a[1:] + [last_action], device=self.dev), 4).float().unsqueeze(0)

    def pooled_next_for_action(self, ai):
        return self.readout(self.m.op_predict_next(self.s, self._a_hist(ai))).squeeze(0)

    def rollout_decode(self, decode_state, action, K):
        s_win, a = self.s, list(self.a)
        s_next = s_win[:, -1]
        for _ in range(max(1, int(K))):
            a_hist = F.one_hot(torch.tensor(a[1:] + [action], device=self.dev), 4).float().unsqueeze(0)
            s_next = self.m.op_predict_next(s_win, a_hist)              # [1,K,D]
            s_win = torch.cat([s_win[:, 1:], s_next.unsqueeze(1)], dim=1)
            a = (a + [action])[-self.W:]
        return decode_state(s_next)[0]

    def peek_frame_slots(self, frame):
        """Slots for a HYPOTHETICAL next frame, computed in the trained window regime
        (replaces the last grid, reruns the W-frame chain). Does not mutate state."""
        z = self.m.encode_frame(frame)
        zwin = torch.stack(self.zg[1:] + [z], dim=1)
        return self.m.slots_of_window(zwin)[:, -1]                      # [1,K,D]


@torch.no_grad()
def gather_slot_pairs(m, frames, positions, actions, starts, tot, W, dev, n_win=3000, episodes=None):
    """Window-harvest RECURRENT slot states for probe fitting: returns
    (S_real [M,K,D], Y_real) = the window's last REAL slot state + its position, and
    (S_pred [M,K,D], Y_pred) = the op-predictor's imagined next slots + the TRUE next position.
    Recurrent depth W matches control-time conditions (probes must be fit in the regime they
    are used). `episodes` restricts sampling to those episode indices -- probe fit/eval MUST be
    split at the EPISODE level (stride-1 windows overlap 6/7 frames; a window-level split leaks
    and inflated slot G1 0.556 vs the honest cross-episode 2.000)."""
    E = starts.shape[0]; rng = np.random.RandomState(1); ss = []
    for _ in range(n_win):
        e = int(episodes[rng.randint(len(episodes))]) if episodes is not None else rng.randint(E - 1)
        s0 = int(starts[e]); end = int(starts[e + 1]) if e + 1 < E else tot
        if end - s0 >= W + 1:
            ss.append(rng.randint(s0, end - W))
    ss = np.array(ss); Sr, Yr, Sp, Yp, bs = [], [], [], [], 48
    for c in range(0, len(ss), bs):
        sb = ss[c:c + bs]
        fidx = np.stack([sb + k for k in range(W + 1)], 1)                    # [B,W+1]
        fr = frames[torch.as_tensor(fidx.reshape(-1))].to(dev).float() / 255.
        z = m.encode_frame(fr); N, D = z.shape[1], z.shape[2]
        s = m.slots_of_window(z.reshape(len(sb), W + 1, N, D))                # [B,W+1,K,D] recurrent
        a_hist = F.one_hot(actions[torch.as_tensor(fidx[:, :W].reshape(-1))].to(dev).long(), 4).float().reshape(len(sb), W, 4)
        s_pred = m.op_predict_next(s[:, :W], a_hist)                          # [B,K,D] imagined next slots
        Sr.append(s[:, W - 1].cpu()); Yr.append(positions[torch.as_tensor(fidx[:, W - 1])])
        Sp.append(s_pred.cpu()); Yp.append(positions[torch.as_tensor(fidx[:, W])])
    return torch.cat(Sr), torch.cat(Yr), torch.cat(Sp), torch.cat(Yp)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--spatial-grid", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--egocentric", action="store_true", help="agent-centered control episodes (match egocentric training)")
    ap.add_argument("--perception-radius", type=float, default=None, help="limited perception disk radius (match training)")
    ap.add_argument("--block-mode", action="store_true", help="Block-Rooms env (match training)")
    ap.add_argument("--block-wall", action="store_true", help="Block-Rooms WALL+GAP variant (match training)")
    ap.add_argument("--block-gate", action="store_true", help="Block-Rooms SWITCH-GATE (key-locked gap; match training)")
    ap.add_argument("--block-radius", type=float, default=None, help="block render radius (MUST match training)")
    ap.add_argument("--block-step-scale", type=float, default=None, help="block step scale (MUST match training)")
    ap.add_argument("--block-clutter", action="store_true", help="CLUTTERED Block-Rooms (match training)")
    ap.add_argument("--n-distractors", type=int, default=4)
    ap.add_argument("--no-bn-calib", action="store_true",
                    help="disable BatchNorm running-stat calibration (debug: reproduces the "
                         "batch-dependent single-frame encoding bug)")
    ap.add_argument("--readout", choices=["ridge", "softargmax", "slot"], default="ridge",
                    help="position readout for control. 'ridge' = grid-pool + linear (cell-quantised, "
                         "overfits at fine grids). 'softargmax' = sub-cell heatmap centroid (~198 params, "
                         "no overfit) -> sharp decode for a SMALL agent among distractors (grid16: 0.10 vs "
                         "ridge 1.11) but reads a DIFFUSE imagination flat. 'slot' = Slot-Attention "
                         "object binding (size-invariant, AGGREGATES the diffuse imagination; the "
                         "readout half of docs/SLOT_FOUR_BRAIN.md).")
    ap.add_argument("--slot-epochs", type=int, default=3000,
                    help="training steps for the slot readout probes (GPU budget; the 400-step CPU "
                         "R1 test under-trained -- real-frame slot decode needs >=2000)")
    ap.add_argument("--num-slots", type=int, default=6, help="number of slots (objects + background)")
    ap.add_argument("--slot-samples", type=int, default=6000, help="token-grid samples for the slot fit")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    m, W = load_model(a.model_path, dev)
    g = a.spatial_grid
    slot_model = getattr(m, "slot_mode", False)
    if slot_model:
        # R2 v2: the control STATE is the model's own object slots, computed EXACTLY as in
        # training (W-frame window chains via SlotHistoryBuffer -- an unbounded episode chain
        # drifts out of the trained regime). readout flattens the K slots for latent-space rows.
        enc = m.encode_frame
        readout = lambda s: s.reshape(*s.shape[:-2], -1)
        print(f"[slot-model] operative control in {m.num_slots}-slot object space (windowed recurrent binding)")
    else:
        enc = m.encode_frame
        readout = lambda z: m.spatial_readout(z, grid=g)
    frames, actions, positions, room_ids, starts = load_raw(a.data_path)
    tot = frames.shape[0]
    if not a.no_bn_calib:
        nb = calibrate_bn(m, frames, dev)
        print(f"[calibrate_bn] populated running stats for {nb} encoder BatchNorm layer(s) "
              f"-> single-frame inference is now batch-independent")
    rng = np.random.RandomState(0); idx = rng.permutation(tot)[:8000]

    def gr(ix, bs=128):
        o = []
        for c in range(0, len(ix), bs):
            b = torch.as_tensor(np.asarray(ix[c:c + bs]))
            o.append(readout(enc(frames[b].to(dev).float() / 255.)).cpu())
        return torch.cat(o)

    tr, va = idx[:6000], idx[6000:8000]
    if slot_model:
        # probes fit on WINDOW-HARVESTED recurrent slot states (the regime control runs in),
        # using the PERMUTATION-EQUIVARIANT probe: slot assignment is arbitrary per episode, so
        # a flat concatenation ridge is permutation-blind and mismeasures by >2x (G1 1.21 flat
        # vs 0.533 equivariant on the same v2 slots). decode_state reads real slots,
        # decode_calib the predictor's imagined next-slots.
        E = starts.shape[0]
        ep_perm = np.random.RandomState(7).permutation(E - 1)
        ep_fit, ep_val = ep_perm[:int(len(ep_perm) * 0.8)], ep_perm[int(len(ep_perm) * 0.8):]
        Sr, Yr, Sp, Yp = gather_slot_pairs(m, frames, positions, actions, starts, tot, W, dev,
                                           n_win=5000, episodes=ep_fit)
        Sv, Yv, _, _ = gather_slot_pairs(m, frames, positions, actions, starts, tot, W, dev,
                                         n_win=1200, episodes=ep_val)
        decode_state = fit_eq_slot_probe(Sr, Yr, dev)
        g1 = (decode_state(Sv) - Yv.to(dev)).norm(dim=1).mean().item()   # HONEST: cross-episode
        decode_calib = fit_eq_slot_probe(Sp, Yp, dev)
    elif a.readout == "slot":
        # object-centric slot readout: size-invariant real-frame decode + a CALIBRATED twin fit
        # on the predictor's own outputs (aggregates the diffuse imagination the peak-centroid
        # readouts read flat -- R1 measured 21x the action_spread of soft-argmax).
        tr_s = tr[:min(len(tr), a.slot_samples)]
        Zt_tr = _gather_token_grids(m, frames, tr_s, dev)
        sa = fit_slot_decode(Zt_tr, positions[torch.as_tensor(np.asarray(tr_s))], dev,
                             num_slots=a.num_slots, epochs=a.slot_epochs)
        del Zt_tr
        Zp, Yp = gather_pred_grids(m, frames, positions, actions, starts, tot, W, dev,
                                   n_win=a.slot_samples)
        sa_c = fit_slot_decode(Zp, Yp, dev, num_slots=a.num_slots, epochs=a.slot_epochs)
        del Zp, Yp
        decode_state = sa                    # goal / current position (real frames)
        decode_calib = sa_c                  # rollout of the predictor's off-manifold output
        g1 = (sa(_gather_token_grids(m, frames, va, dev)) - positions[torch.as_tensor(va)].to(dev)).norm(dim=1).mean().item()
    elif a.readout == "softargmax":
        # sub-cell heatmap centroid: sharp for a SMALL agent (grid16 G1 ~0.10 vs ridge ~1.11),
        # no overfit (the fine-grid ridge is 49k-dim and OOMs/overfits). decode_state reads the
        # token grid directly. The calibrated path reuses it (soft-argmax is already on-manifold).
        sa = fit_softargmax_decode(m, frames, positions, tr, dev)                 # reads REAL frames
        sa_c = fit_calibrated_softargmax(m, frames, positions, actions, starts, tot, W, dev)  # reads IMAGINED latents
        decode_state = sa                    # goal / current position (real frames)
        decode_calib = sa_c                  # rollout of the predictor's off-manifold output
        g1 = (sa(_gather_token_grids(m, frames, va, dev)) - positions[torch.as_tensor(va)].to(dev)).norm(dim=1).mean().item()
    else:
        ridge = fit_ridge_decode(gr(tr), positions[torch.as_tensor(tr)], dev)
        decode_state = lambda grid: ridge(m.spatial_readout(grid, grid=g))
        g1 = (ridge(gr(va)) - positions[torch.as_tensor(va)].to(dev)).norm(dim=1).mean().item()
        ridge_c = fit_calibrated_decode(m, frames, positions, actions, starts, tot, readout, W, dev)
        decode_calib = lambda grid: ridge_c(m.spatial_readout(grid, grid=g))
    # decoded-state forward-dynamics probe uses the token ridge; skip it in softargmax/slot/slot-model modes
    fwd = None if (slot_model or a.readout in ("softargmax", "slot")) else \
        fit_forward_probe(m, frames, positions, actions, starts, tot, readout, ridge, dev)

    spreads, errs, dir_hits = [], [], 0
    spreads_c, errs_c, dir_hits_c = [], [], 0
    errs_f, dir_hits_f = [], 0
    errs_od, dir_hits_od = [], 0           # ORACLE-DECODE: decode the REAL next frame (instrument check)
    lat_dir_hits = 0                       # LeWM-native forward latent-space control
    inv_dir_hits = 0                       # INVERSE-dynamics goal-emission control (option 2)
    imag_err, ro_scale = [], []            # predictor 1-step imagination accuracy (latent)
    n, ep = 0, 0
    while n < a.n_steps:
        env = TwoRoomsEnv(seed=3000 + ep, complex_mode=False, hazards=False, egocentric=a.egocentric,
                          perception_radius=a.perception_radius, block_mode=a.block_mode, block_wall=a.block_wall, block_gate=a.block_gate,
                          block_radius=a.block_radius, block_step_scale=a.block_step_scale,
                          block_clutter=a.block_clutter, n_distractors=a.n_distractors)
        obs = env.reset(start_room=0, goal_room=1); ep += 1
        target = obs["target"]
        # GOAL latent: the view with the agent AT the target (LeWM-native control compares the
        # predicted next latent to THIS, with NO decoding -> immune to the off-manifold decode).
        eg = TwoRoomsEnv(seed=3000 + ep, complex_mode=False, hazards=False, egocentric=a.egocentric,
                         perception_radius=a.perception_radius, block_mode=a.block_mode, block_wall=a.block_wall, block_gate=a.block_gate,
                          block_radius=a.block_radius, block_step_scale=a.block_step_scale,
                          block_clutter=a.block_clutter, n_distractors=a.n_distractors)
        eg.reset(start_room=0, goal_room=1); eg.agent_pos = target.copy(); eg.target_pos = target.copy()
        if slot_model:
            buf = SlotHistoryBuffer(m, W, dev, readout); buf.reset(obs_to_frame(obs, dev))
            enc_peek = buf.peek_frame_slots                    # counterfactuals in the trained window regime
            gz = m.encode_frame(obs_to_frame({"image": eg.render()}, dev).unsqueeze(0))
            goal_z = m.slots_of_window(torch.stack([gz] * W, dim=1))[:, -1]   # goal slots, same regime
        else:
            enc_peek = enc
            goal_z = enc_peek(obs_to_frame({"image": eg.render()}, dev).unsqueeze(0))
            buf = HistoryBuffer(m, W, dev, readout=readout, encode=enc)
            buf.reset(obs_to_frame(obs, dev))
        goal_ro = readout(goal_z).squeeze(0)
        goal_pool = m.pool(goal_z).squeeze(0)                 # pooled goal state for inverse dynamics
        for _ in range(40):
            if n >= a.n_steps:
                break
            pred = np.stack([buf.rollout_decode(decode_state, ai, 1).cpu().numpy() for ai in range(4)])   # [4,2]
            pred_c = np.stack([buf.rollout_decode(decode_calib, ai, 1).cpu().numpy() for ai in range(4)])  # [4,2]
            nxt = [copy.deepcopy(env).step(ai)[0] for ai in range(4)]
            true = np.stack([o["position"] for o in nxt])                                                  # [4,2]
            ta = int(np.argmin(np.linalg.norm(true - target, axis=1)))
            # ORACLE-DECODE: decode the REAL next frame for each action (NOT the predictor). This
            # isolates the decode+instrument chain from the predictor: with G1<<step it MUST rank
            # actions (direction_acc ~1.0). If it does not, the failure is the control-loop
            # decode/readout/env (not the predictor); if it does, the predictor is the sole problem.
            pred_od = np.stack([decode_state(
                enc_peek(obs_to_frame({"image": o["image"]}, dev).unsqueeze(0)))[0].cpu().numpy()
                for o in nxt])                                                                             # [4,2]
            errs_od.append(float(np.linalg.norm(pred_od - true, axis=1).mean()))
            dir_hits_od += int(int(np.argmin(np.linalg.norm(pred_od - target, axis=1))) == ta)
            spreads.append(float(pred.std(0).mean())); errs.append(float(np.linalg.norm(pred - true, axis=1).mean()))
            dir_hits += int(int(np.argmin(np.linalg.norm(pred - target, axis=1))) == ta)
            spreads_c.append(float(pred_c.std(0).mean())); errs_c.append(float(np.linalg.norm(pred_c - true, axis=1).mean()))
            dir_hits_c += int(int(np.argmin(np.linalg.norm(pred_c - target, axis=1))) == ta)
            if fwd is not None:
                dp_cur = decode_state(buf.cur_z)                                                           # decoded current pos [1,2]
                pred_f = np.stack([fwd(dp_cur, ai)[0].cpu().numpy() for ai in range(4)])                   # [4,2]
                errs_f.append(float(np.linalg.norm(pred_f - true, axis=1).mean()))
                dir_hits_f += int(int(np.argmin(np.linalg.norm(pred_f - target, axis=1))) == ta)
            lat = [float((buf.pooled_next_for_action(ai) - goal_ro).norm()) for ai in range(4)]   # NO decode
            lat_dir_hits += int(int(np.argmin(lat)) == ta)
            # INVERSE-dynamics goal-emission (option 2): ask "which action moves me toward the goal?"
            cur_pool = m.pool(buf.cur_z).squeeze(0)
            inv_logits = m.inverse_action(cur_pool.unsqueeze(0), goal_pool.unsqueeze(0))[0]      # [4]
            inv_dir_hits += int(int(inv_logits.argmax()) == ta)
            # predictor 1-step IMAGINATION accuracy (latent, for the action actually taken): does the
            # imagined next latent match the TRUE next latent? limited perception should sharpen this.
            tn = copy.deepcopy(env).step(ta)[0]
            true_ro = readout(enc_peek(obs_to_frame({"image": tn["image"]}, dev).unsqueeze(0))).squeeze(0)
            imag_err.append(float((buf.pooled_next_for_action(ta) - true_ro).norm()))
            ro_scale.append(float(true_ro.norm()))
            n += 1
            obs, _, done, info = env.step(ta); buf.push(obs_to_frame(obs, dev), ta)  # traverse via TRUE best
            if done or info["distance"] < 0.6:
                break

    print(f"G1_spatial(real frames) = {g1:.3f} wu")
    print(f"--- ORACLE-DECODE: decode the REAL next frame per action (INSTRUMENT CHECK, no predictor) ---")
    print(f"  pred_err {np.mean(errs_od):.3f} | direction_acc {dir_hits_od/max(1,n):.2f}   "
          f"<<< MUST be ~1.0 if decode resolves the step; else the control-loop decode is broken")
    print(f"--- decode fit on REAL frames (current control path) ---")
    print(f"  action_spread {np.mean(spreads):.3f} | pred_err {np.mean(errs):.3f} | direction_acc {dir_hits/max(1,n):.2f}")
    print(f"--- decode CALIBRATED on op-predictor outputs ---")
    print(f"  action_spread {np.mean(spreads_c):.3f} | pred_err {np.mean(errs_c):.3f} | direction_acc {dir_hits_c/max(1,n):.2f}")
    print(f"--- FORWARD-DYNAMICS PROBE  g(latent,action)->next pos  (decoded-state) ---")
    if fwd is None:
        print(f"  (skipped in --readout {a.readout} mode)")
    else:
        print(f"  pred_err {np.mean(errs_f):.3f} | direction_acc {dir_hits_f/max(1,n):.2f}")
    print(f"--- LATENT-SPACE control (forward: nearest predicted latent to GOAL latent, NO decode) ---")
    print(f"  direction_acc {lat_dir_hits/max(1,n):.2f}")
    print(f"--- INVERSE-DYNAMICS goal-emission (option 2: inv_head(current, goal) -> action) ---")
    print(f"  direction_acc {inv_dir_hits/max(1,n):.2f}   <<< uses the part of the model that IS accurate (inv 0.03)")
    _ie, _sc = np.mean(imag_err), np.mean(ro_scale)
    print(f"--- PREDICTOR 1-STEP IMAGINATION accuracy (latent) ---")
    print(f"  imag_err {_ie:.2f} / readout_norm {_sc:.2f} = {_ie/max(1e-6,_sc):.3f} relative   "
          f"<<< lower = the predictor imagines the next latent accurately (limited perception should drop this)")
    print(f"[direction_acc: 1.0 perfect, 0.25 random; >0.6 => USABLE for greedy control]")


if __name__ == "__main__":
    main()
