"""(b) Does the op-predictor LEARN that an into-wall action is BLOCKED? For data transitions
where the agent is adjacent to the wall (x in [4.4,5.0], not at the door) and takes the
into-wall action (right), compare the predictor's IMAGINED next x-position vs the TRUE next
x-position. If the predictor imagines moving THROUGH the wall (pred dx ~ +0.3) while the truth
is blocked (true dx ~ 0), it is wall-blind -> the control walks into the wall. Baseline: the
same agent taking an into-wall action while AT the door (should pass -> both dx>0).
Usage: python predictor_wall_test.py <data.pt> <model.pt>
"""
import sys; sys.path.insert(0, "src")
import numpy as np, torch, argparse, torch.nn.functional as F
from alps.training.train_hier import load_raw
from alps.evaluation.validate_temporal import (load_model, calibrate_bn, gather_pred_grids,
                                               fit_ridge_decode)

ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("model")
ap.add_argument("--grid", type=int, default=8); a = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.set_grad_enabled(False)
m, W = load_model(a.model, dev)
frames, actions, positions, room_ids, starts = load_raw(a.data)
tot = frames.shape[0]; calibrate_bn(m, frames, dev)
readout = lambda z: m.spatial_readout(z, grid=a.grid)
# A3 calibrated decode (predictor outputs -> position), for reading imagination honestly
Zp, Yp = gather_pred_grids(m, frames, positions, actions, starts, tot, W, dev, n_win=3000)
Xp = torch.cat([readout(Zp[c:c+128].to(dev)).detach().cpu().float() for c in range(0, len(Zp), 128)])
calib = fit_ridge_decode(Xp, Yp, dev); decode = lambda grid: calib(readout(grid))

WALLX, DOOR = 5.0, (4.5, 5.5)
st = list(starts) + [tot]; ep_of = np.zeros(tot, int)
for e in range(len(st) - 1):
    ep_of[st[e]:st[e+1]] = e

def collect(cond_name, mask_fn):
    """windows [t-W+1..t] where frame t satisfies mask_fn and action t is into-wall(=3); predict t+1."""
    ts = []
    for t in range(tot - 1):
        if t + 1 >= tot or ep_of[t+1] != ep_of[t]: continue
        if t - W + 1 < st[ep_of[t]]: continue
        x, y = positions[t];
        if int(actions[t]) != 3: continue                 # right = into the wall from the left
        if not mask_fn(float(x), float(y)): continue
        ts.append(t)
    if not ts: print(f"  [{cond_name}] no matching transitions"); return
    ts = np.array(ts[:2000]); pdx, tdx = [], []
    for c in range(0, len(ts), 64):
        tb = ts[c:c+64]
        fidx = np.stack([tb + k for k in range(-W+1, 1)], 1)   # [B,W]
        fr = frames[torch.as_tensor(fidx.reshape(-1))].to(dev).float()/255.
        z = m.encode_frame(fr).reshape(len(tb), W, -1, m.d_model)
        a_hist = F.one_hot(actions[torch.as_tensor(fidx)].long(), 4).float().to(dev)
        zp = m.op_predict_next(z, a_hist)                       # imagined next grid
        pnext = decode(zp).cpu().numpy(); cur = decode(z[:, -1]).cpu().numpy()
        pdx.extend((pnext[:, 0] - cur[:, 0]).tolist())
        tdx.extend((positions[torch.as_tensor(tb+1)][:, 0].numpy() - positions[torch.as_tensor(tb)][:, 0].numpy()).tolist())
    pdx, tdx = np.array(pdx), np.array(tdx)
    print(f"  [{cond_name}] n={len(pdx)}")
    print(f"     TRUE  next dx (right action): mean {tdx.mean():+.3f}  (blocked~0, moved~+0.3)")
    print(f"     PRED  next dx (imagined)    : mean {pdx.mean():+.3f}")
    print(f"     -> predictor imagines MOVING THROUGH on {100*(pdx>0.12).mean():.0f}% of blocked cases"
          f" (true blocked {100*(tdx<0.08).mean():.0f}%)")

print("=== INTO-WALL, agent adjacent to wall, NOT at door (should be BLOCKED) ===")
collect("at-wall/into-wall", lambda x, y: 4.4 <= x <= 5.0 and not (DOOR[0] <= y <= DOOR[1]))
print("=== INTO-WALL, agent adjacent to wall, AT the door (should PASS) ===")
collect("at-door/into-wall", lambda x, y: 4.4 <= x <= 5.0 and DOOR[0] <= y <= DOOR[1])
