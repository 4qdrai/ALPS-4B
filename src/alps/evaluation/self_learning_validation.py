"""
ALPS-4B — honest validation of the Latent-RAG "no-retraining" self-learning loop.

The shipped self_learning_demo recalls the exact vector it just wrote, so its
">98% error reduction" is trivially true on the stored point and proves nothing
about learning. This module runs the test that actually matters, with proper
splits:

  Surprise set  = the transitions the trained world model predicts WORST
                  (top-error quantile) — i.e. a novel/under-modeled regime.
  WRITE split   = half of the surprise set; Delta z = (z_next - z_pred) is
                  written to the Latent-RAG cache keyed by the pooled context.
  TEST split    = the OTHER half of the surprise set (never written).
                  -> measures GENERALIZATION (the missing test).
  CONTROL split = transitions the model already predicts WELL (low-error).
                  -> measures INTERFERENCE (writing must not harm these).

For each split we report mean latent prediction error WITHOUT and WITH RAG
retrieval, plus the % reduction. A credible self-learning claim needs:
  * large reduction on WRITE (one-shot recall),
  * positive reduction on TEST (generalization to unseen similar contexts),
  * near-zero / non-negative change on CONTROL (no interference).

USAGE
    PYTHONPATH=src python -m alps.evaluation.self_learning_validation \
        --model-path results/two_rooms/validation/repr_world_model_fs4.pt
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, "src")

import argparse, json
import numpy as np
import torch
import torch.nn.functional as F

from alps.benchmarks.two_rooms.dataset import TwoRoomsDataset
from alps.core.latent_rag import LatentRAG
from alps.evaluation.repr_decoder_gate import ReprWorldModel, split_dataset


@torch.no_grad()
def gather_transitions(model, dataset, idx, device, max_n=4000):
    """Return pooled context, z_t, z_next, z_pred, per-sample error."""
    ctx, ZT, ZN, ZP, ERR = [], [], [], [], []
    n = 0
    for i in idx:
        s = dataset[i]
        fr = s["video_frames"].to(device)
        acts = s["actions_onehot"].to(device)
        T = fr.shape[1]
        zs = [model.encode_frame(fr[:, t].unsqueeze(0)) for t in range(T)]
        for t in range(T - 1):
            zt, zn = zs[t], zs[t + 1]
            zp = model.predict_next(zt, acts[t].unsqueeze(0))
            err = (zp - zn).flatten(1).norm(dim=1)
            ctx.append(zt.mean(1).squeeze(0).cpu())
            ZT.append(zt.squeeze(0).cpu()); ZN.append(zn.squeeze(0).cpu())
            ZP.append(zp.squeeze(0).cpu()); ERR.append(err.item())
            n += 1
        if n >= max_n:
            break
    return (torch.stack(ctx), torch.stack(ZT), torch.stack(ZN),
            torch.stack(ZP), torch.tensor(ERR))


def run(args):
    device = torch.device(args.device)
    dataset = TwoRoomsDataset(args.data_path, clip_length=8, stride=4, frame_skip=args.frame_skip)
    if args.limit_clips:
        dataset.clip_indices = dataset.clip_indices[: args.limit_clips]
    _, val_idx = split_dataset(dataset, val_frac=0.4, seed=0)

    ckpt = torch.load(args.model_path, map_location=device, weights_only=True)
    model = ReprWorldModel(d_model=ckpt.get("d_model", 128)).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()

    ctx, ZT, ZN, ZP, ERR = gather_transitions(model, dataset, val_idx, device)
    print(f"[data] {len(ERR)} held-out transitions | mean err {ERR.mean():.3f}")

    # surprise = top-error quantile; control = bottom-error quantile
    q_hi = torch.quantile(ERR, 1 - args.surprise_frac)
    q_lo = torch.quantile(ERR, args.surprise_frac)
    surprise = torch.where(ERR >= q_hi)[0]
    control = torch.where(ERR <= q_lo)[0]
    perm = surprise[torch.randperm(len(surprise))]
    write_ids = perm[: len(perm) // 2]
    test_ids = perm[len(perm) // 2:]
    print(f"[split] write={len(write_ids)} test={len(test_ids)} control={len(control)}")

    rag = LatentRAG(d_model=model.d_model, sim_threshold=args.sim_threshold,
                    max_size=max(5000, len(write_ids) + 10)).to(device)

    # write Delta z corrections for the WRITE split
    for i in write_ids.tolist():
        key = ctx[i].to(device)
        delta = (ZN[i] - ZP[i]).mean(0).to(device)   # pooled correction [D]
        rag.write_memory(key, delta)

    def err_with_rag(ids):
        base, corr = [], []
        for i in ids.tolist():
            zp = ZP[i].to(device)                     # [N,D]
            zn = ZN[i].to(device)
            base.append((zp - zn).flatten().norm().item())
            q = ctx[i].to(device).view(1, 1, -1)      # [1,1,D]
            d = rag.retrieve_correction(q).squeeze(0)  # [1,D]
            zp_corr = zp + d                           # broadcast [N,D]
            corr.append((zp_corr - zn).flatten().norm().item())
        return float(np.mean(base)), float(np.mean(corr))

    out = {}
    for name, ids in [("write_oneshot", write_ids), ("test_generalization", test_ids),
                      ("control_interference", control)]:
        b, c = err_with_rag(ids)
        red = 100.0 * (b - c) / max(b, 1e-8)
        out[name] = {"n": int(len(ids)), "err_no_rag": b, "err_with_rag": c, "reduction_pct": red}
        print(f"  {name:22s}: no-RAG {b:.3f} -> RAG {c:.3f}  ({red:+.1f}%)")

    verdict = {
        "oneshot_ok": out["write_oneshot"]["reduction_pct"] > 50,
        "generalizes": out["test_generalization"]["reduction_pct"] > 5,
        "no_interference": out["control_interference"]["reduction_pct"] > -5,
    }
    out["verdict"] = verdict
    out["pass"] = all(verdict.values())
    os.makedirs(args.save_dir, exist_ok=True)
    p = os.path.join(args.save_dir, "self_learning_validation.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[verdict] {verdict} -> {'PASS' if out['pass'] else 'NEEDS WORK'}")
    print(f"[report] {p}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="results/two_rooms/validation/repr_world_model_fs4.pt")
    ap.add_argument("--data-path", default="data/two_rooms/trajectories.pt")
    ap.add_argument("--save-dir", default="results/two_rooms/validation")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--frame-skip", type=int, default=4)
    ap.add_argument("--surprise-frac", type=float, default=0.25)
    ap.add_argument("--sim-threshold", type=float, default=0.6)
    ap.add_argument("--limit-clips", type=int, default=0)
    main_args = ap.parse_args()
    run(main_args)


if __name__ == "__main__":
    main()
