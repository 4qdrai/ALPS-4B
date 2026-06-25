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

from alps.benchmarks.two_rooms.environment import TwoRoomsEnv
from alps.benchmarks.two_rooms.world_model_planning import obs_to_frame
from alps.training.train_hier import load_raw
from alps.evaluation.validate_temporal import load_model, fit_ridge_decode, HistoryBuffer


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--spatial-grid", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=200)
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

    spreads, errs, dir_hits, n, ep = [], [], 0, 0, 0
    while n < a.n_steps:
        env = TwoRoomsEnv(seed=3000 + ep, complex_mode=False, hazards=False)
        obs = env.reset(start_room=0, goal_room=1); ep += 1
        target = obs["target"]
        buf = HistoryBuffer(m, W, dev, readout=readout); buf.reset(obs_to_frame(obs, dev))
        for _ in range(40):
            if n >= a.n_steps:
                break
            pred = np.stack([buf.rollout_decode(decode_state, ai, 1).cpu().numpy() for ai in range(4)])  # [4,2]
            true = np.stack([copy.deepcopy(env).step(ai)[0]["position"] for ai in range(4)])              # [4,2]
            spreads.append(float(pred.std(0).mean()))
            errs.append(float(np.linalg.norm(pred - true, axis=1).mean()))
            pa = int(np.argmin(np.linalg.norm(pred - target, axis=1)))
            ta = int(np.argmin(np.linalg.norm(true - target, axis=1)))
            dir_hits += int(pa == ta); n += 1
            obs, _, done, info = env.step(ta); buf.push(obs_to_frame(obs, dev), ta)  # traverse via TRUE best
            if done or info["distance"] < 0.6:
                break

    print(f"G1_spatial(real frames) = {g1:.3f} wu")
    print(f"action_spread           = {np.mean(spreads):.3f} wu   "
          f"[~0 => predictor IGNORES action; healthy ~0.5-1.0]")
    print(f"pred_err (pred vs TRUE)  = {np.mean(errs):.3f} wu")
    print(f"direction_acc           = {dir_hits / max(1, n):.2f}      "
          f"[1.0 perfect, 0.25 random; <0.5 => predictor unusable for greedy control]")


if __name__ == "__main__":
    main()
