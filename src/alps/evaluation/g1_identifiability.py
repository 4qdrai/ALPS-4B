"""
G1 LINEAR IDENTIFIABILITY — self-supervised vs supervised, head-to-head.

Trains TWO temporal encoders on the SAME data/config:
  * SSL  — LeWM-faithful PURE self-supervised: the encoder learns ONLY from feature
           prediction (predict z_{t+1} from z_{<=t}, a_t) + SIGReg/VICReg collapse
           prevention. NO position or dynamics labels ever touch the encoder.
  * SUP  — position-grounded: the same model with the position/dynamics aux losses ON
           (pos_weight, dyn_weight > 0), so labels DO shape the encoder.

Then, for EACH frozen encoder, it fits a FRESH linear probe on the frozen latents
(standard JEPA frozen-probe protocol) and reports:
  * G1   — held-out mean Euclidean position-decode error (world units),
  * collapse diagnostics — effective rank / dead dims / mean pairwise cosine.

The headline is the GAP `G1_ssl - G1_sup`: LeWM's "linear identifiability" claim is
that a purely self-supervised latent already encodes state linearly, so the gap is
small and G1_ssl stays well under the planning threshold (~0.3 wu). This is the gate
that says the frozen-latent + post-hoc-probe stack rides on an UNSUPERVISED
representation, not one that was told where the agent is.

USAGE
  PYTHONPATH=src python -m alps.evaluation.g1_identifiability \
      --data-path data/two_rooms/trajectories.pt --epochs 40 \
      --limit-samples 6000 [--save-models]
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, "src")
import argparse, json, copy
import numpy as np
import torch

from alps.training.train_temporal import train, build_parser
from alps.training.train_hier import load_raw
from alps.evaluation.validate_temporal import gather
from alps.evaluation.validate_hierarchy import fit_probe


@torch.no_grad()
def measure(model, frames, positions, room_ids, device, limit_samples, seed=1):
    """Freeze the encoder, fit a fresh position probe on frozen latents, report
    held-out G1 + latent-collapse diagnostics."""
    model.eval()
    total = frames.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(total)[:limit_samples] if limit_samples else rng.permutation(total)
    ntr = int(len(idx) * 0.8); tr, va = idx[:ntr], idx[ntr:]
    Ztr, _, _, _, Ptr = gather(model, frames, positions, room_ids, tr, device)
    Zva, _, _, _, Pva = gather(model, frames, positions, room_ids, va, device)
    torch.set_grad_enabled(True)
    decode = fit_probe(Ztr, Ptr, device)
    torch.set_grad_enabled(False)
    g1 = (decode(Zva.to(device)) - Pva.to(device)).norm(dim=1).mean().item()
    # collapse diagnostics
    Zc = Zva - Zva.mean(0)
    ev = torch.linalg.eigvalsh((Zc.t() @ Zc) / (Zva.shape[0] - 1)).clamp(min=0)
    eff_rank = float((ev.sum() ** 2 / (ev ** 2).sum()))
    std = Zva.std(0)
    zn = torch.nn.functional.normalize(Zva[:600], dim=1)
    cos = zn @ zn.t()
    cos_mean = float(cos[~torch.eye(zn.shape[0], dtype=torch.bool)].mean())
    return {
        "G1_decode_err_world_units": g1, "passed": g1 < 0.3,
        "d_model": int(Zva.shape[1]), "effective_rank": eff_rank,
        "dead_dims": int((std < 0.01).sum()), "mean_pairwise_cosine": cos_mean,
        "catastrophic_collapse": bool(eff_rank < 1.5 or cos_mean > 0.99),
    }


def make_args(base, self_supervised, out):
    a = copy.deepcopy(base)
    a.self_supervised = self_supervised
    a.out = out
    return a


def run(args):
    device = torch.device(args.device)
    # base training args = train_temporal defaults, overridden by this CLI
    base = build_parser().parse_args([])
    for k in ("data_path", "device", "epochs", "batch_size", "lr", "d_model", "enc_depth",
              "enc_heads", "op_depth", "abs_depth", "num_codes", "num_experts",
              "active_experts", "window", "stride", "sample_stride", "limit_samples",
              "collapse_weight", "lambda_sigreg", "sigreg_slices"):
        if hasattr(args, k) and getattr(args, k) is not None:
            setattr(base, k, getattr(args, k))
    base.save_model = args.save_models

    frames, _, positions, room_ids, _ = load_raw(args.data_path)
    out = {}
    for tag, ssl in (("SUP", False), ("SSL", True)):
        print(f"\n================ training {tag} encoder (self_supervised={ssl}) ================")
        mpath = os.path.join(args.out_dir, f"temporal_{tag.lower()}.pt")
        model = train(make_args(base, ssl, mpath))
        out[tag] = measure(model, frames, positions, room_ids, device,
                           args.eval_samples, seed=1)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    gap = out["SSL"]["G1_decode_err_world_units"] - out["SUP"]["G1_decode_err_world_units"]
    out["G1_gap_ssl_minus_sup"] = gap
    os.makedirs(args.out_dir, exist_ok=True)
    p = os.path.join(args.out_dir, "g1_identifiability.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=float)

    def row(t):
        d = out[t]
        return (f"  {t}: G1 {d['G1_decode_err_world_units']:.3f}wu "
                f"({'PASS' if d['passed'] else 'FAIL'}) | eff-rank {d['effective_rank']:.1f}/{d['d_model']} "
                f"| dead {d['dead_dims']} | cos {d['mean_pairwise_cosine']:.2f} "
                f"| {'COLLAPSE' if d['catastrophic_collapse'] else 'ok'}")
    print("\n===== G1 LINEAR IDENTIFIABILITY (frozen encoder + fresh probe) =====")
    print(row("SUP")); print(row("SSL"))
    print(f"  GAP  G1_ssl - G1_sup = {gap:+.3f}wu  "
          f"-> {'LeWM-confirmed (small gap, SSL decodes)' if out['SSL']['passed'] else 'SSL does NOT linearly identify position'}")
    print(f"[report] {p}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="data/two_rooms/trajectories.pt")
    ap.add_argument("--out-dir", default="results/two_rooms/validation")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--enc-depth", type=int, default=10)
    ap.add_argument("--enc-heads", type=int, default=8)
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--num-codes", type=int, default=64)
    ap.add_argument("--limit-samples", type=int, default=0, help="cap TRAIN windows (0=all)")
    ap.add_argument("--eval-samples", type=int, default=6000, help="frames for the probe/G1 eval")
    ap.add_argument("--save-models", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
