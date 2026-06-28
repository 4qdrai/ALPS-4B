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
from alps.evaluation.validate_temporal import load_model, fit_ridge_decode, HistoryBuffer


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
    ss = np.array(ss); Xs, Ys, bs = [], [], 64
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


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--spatial-grid", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--egocentric", action="store_true", help="agent-centered control episodes (match egocentric training)")
    ap.add_argument("--perception-radius", type=float, default=None, help="limited perception disk radius (match training)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    m, W = load_model(a.model_path, dev)
    g = a.spatial_grid
    readout = lambda z: m.spatial_readout(z, grid=g)
    frames, actions, positions, room_ids, starts = load_raw(a.data_path)
    tot = frames.shape[0]
    rng = np.random.RandomState(0); idx = rng.permutation(tot)[:8000]

    def gr(ix, bs=128):
        o = []
        for c in range(0, len(ix), bs):
            b = torch.as_tensor(np.asarray(ix[c:c + bs]))
            o.append(readout(m.encode_frame(frames[b].to(dev).float() / 255.)).cpu())
        return torch.cat(o)

    tr, va = idx[:6000], idx[6000:8000]
    ridge = fit_ridge_decode(gr(tr), positions[torch.as_tensor(tr)], dev)
    decode_state = lambda grid: ridge(m.spatial_readout(grid, grid=g))
    g1 = (ridge(gr(va)) - positions[torch.as_tensor(va)].to(dev)).norm(dim=1).mean().item()
    ridge_c = fit_calibrated_decode(m, frames, positions, actions, starts, tot, readout, W, dev)
    decode_calib = lambda grid: ridge_c(m.spatial_readout(grid, grid=g))
    fwd = fit_forward_probe(m, frames, positions, actions, starts, tot, readout, ridge, dev)

    spreads, errs, dir_hits = [], [], 0
    spreads_c, errs_c, dir_hits_c = [], [], 0
    errs_f, dir_hits_f = [], 0
    lat_dir_hits = 0                       # LeWM-native forward latent-space control
    inv_dir_hits = 0                       # INVERSE-dynamics goal-emission control (option 2)
    imag_err, ro_scale = [], []            # predictor 1-step imagination accuracy (latent)
    n, ep = 0, 0
    while n < a.n_steps:
        env = TwoRoomsEnv(seed=3000 + ep, complex_mode=False, hazards=False, egocentric=a.egocentric,
                          perception_radius=a.perception_radius)
        obs = env.reset(start_room=0, goal_room=1); ep += 1
        target = obs["target"]
        # GOAL latent: the view with the agent AT the target (LeWM-native control compares the
        # predicted next latent to THIS, with NO decoding -> immune to the off-manifold decode).
        eg = TwoRoomsEnv(seed=3000 + ep, complex_mode=False, hazards=False, egocentric=a.egocentric,
                         perception_radius=a.perception_radius)
        eg.reset(start_room=0, goal_room=1); eg.agent_pos = target.copy(); eg.target_pos = target.copy()
        goal_z = m.encode_frame(obs_to_frame({"image": eg.render()}, dev).unsqueeze(0))
        goal_ro = readout(goal_z).squeeze(0)
        goal_pool = m.pool(goal_z).squeeze(0)                 # pooled goal latent for inverse dynamics
        buf = HistoryBuffer(m, W, dev, readout=readout); buf.reset(obs_to_frame(obs, dev))
        for _ in range(40):
            if n >= a.n_steps:
                break
            pred = np.stack([buf.rollout_decode(decode_state, ai, 1).cpu().numpy() for ai in range(4)])   # [4,2]
            pred_c = np.stack([buf.rollout_decode(decode_calib, ai, 1).cpu().numpy() for ai in range(4)])  # [4,2]
            true = np.stack([copy.deepcopy(env).step(ai)[0]["position"] for ai in range(4)])               # [4,2]
            ta = int(np.argmin(np.linalg.norm(true - target, axis=1)))
            spreads.append(float(pred.std(0).mean())); errs.append(float(np.linalg.norm(pred - true, axis=1).mean()))
            dir_hits += int(int(np.argmin(np.linalg.norm(pred - target, axis=1))) == ta)
            spreads_c.append(float(pred_c.std(0).mean())); errs_c.append(float(np.linalg.norm(pred_c - true, axis=1).mean()))
            dir_hits_c += int(int(np.argmin(np.linalg.norm(pred_c - target, axis=1))) == ta)
            dp_cur = decode_state(buf.cur_z)                                                               # decoded current pos [1,2]
            pred_f = np.stack([fwd(dp_cur, ai)[0].cpu().numpy() for ai in range(4)])                       # [4,2]
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
            true_ro = readout(m.encode_frame(obs_to_frame({"image": tn["image"]}, dev).unsqueeze(0))).squeeze(0)
            imag_err.append(float((buf.pooled_next_for_action(ta) - true_ro).norm()))
            ro_scale.append(float(true_ro.norm()))
            n += 1
            obs, _, done, info = env.step(ta); buf.push(obs_to_frame(obs, dev), ta)  # traverse via TRUE best
            if done or info["distance"] < 0.6:
                break

    print(f"G1_spatial(real frames) = {g1:.3f} wu")
    print(f"--- decode fit on REAL frames (current control path) ---")
    print(f"  action_spread {np.mean(spreads):.3f} | pred_err {np.mean(errs):.3f} | direction_acc {dir_hits/max(1,n):.2f}")
    print(f"--- decode CALIBRATED on op-predictor outputs ---")
    print(f"  action_spread {np.mean(spreads_c):.3f} | pred_err {np.mean(errs_c):.3f} | direction_acc {dir_hits_c/max(1,n):.2f}")
    print(f"--- FORWARD-DYNAMICS PROBE  g(latent,action)->next pos  (decoded-state) ---")
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
