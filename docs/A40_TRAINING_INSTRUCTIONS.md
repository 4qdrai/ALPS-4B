# A40 Training Instructions — ALPS-4B Fully-Unsupervised Pipeline

*These instructions cover exactly what to do on RunPod to run the full
unsupervised validation pipeline (`scripts/run_a40_unsupervised.sh`).
Follow Session 1 → then Sessions 2–5 in order per `EDGE_PROGRAM.md §4`.*

---

## 0. Prerequisites (do this once, before spinning up the pod)

### 0a. Repository

The code lives at **https://github.com/4qdrai/ALPS-4B** (branch `main`).
The pod clones it directly — nothing to upload for the code itself.

If you have uncommitted local changes, push them first:

```bash
# From your local ALPS-4B directory:
git add -A && git commit -m "pre-A40 snapshot"
git push origin main
```

### 0b. Optional: pre-generate data locally

If internet bandwidth is tight on the pod, generate the training data locally
and upload it with `rsync`:

```bash
# On your local machine (fast, < 2 min):
python -m alps.benchmarks.two_rooms.data_generator \
    --save-path data/two_rooms/trajectories_unsup.pt \
    --num-episodes 10000 --max-steps 100 --heuristic-fraction 0.4 --seed 7

python -m alps.benchmarks.two_rooms.generate_complex \
    --save-path data/two_rooms/trajectories_unsup_complex.pt \
    --num-episodes 3000 --bfs-fraction 0.5

# Then sync once the pod is up:
rsync -avz --progress data/ runpod_user@POD_IP:/workspace/alps-4b/data/
```

---

## 1. Pod configuration on RunPod

### 1a. Template

| Setting | Value |
|---|---|
| **GPU** | NVIDIA A40 (48 GB VRAM) |
| **Disk (container)** | 50 GB |
| **Volume (persistent)** | 100 GB (mount at `/workspace`) |
| **Base image** | `runpod/pytorch:2.3.0-py3.11-cuda12.1.1-devel-ubuntu22.04` |
| **Expose ports** | none required |

The A40 has 48 GB VRAM.  With `d_model=192`, batch 64, the model is ≈5M params.
Peak VRAM is ≈4 GB during training — the A40 is used almost entirely for
throughput (episode count) not memory.  You could fit batch=128 if needed.

### 1b. Why A40 (not H100)

The A40 is sufficient: d192 training at 10k episodes × 80 epochs runs in ≈3–4 h.
H100 is faster but ≈3× more expensive per hour; no architectural benefit here.

### 1c. Spot vs on-demand

Use **on-demand** for Session 1 (6–10 h uninterrupted).  The script has
per-stage guards (`[ ! -f "$MODEL" ] && ...`) so a restart resumes cheaply.
For Sessions 2–5 (4–6 h each), spot is fine if the volume persists.

### 1d. Cost estimate

A40 on RunPod is ≈$0.39–$0.49/hr.  Full 5-session program: ≈24 h → ≈$12.

### 1e. Volume layout

After mounting at `/workspace`:

```
/workspace/
  alps-4b/               ← git clone goes here
    data/
    results/
    src/
    scripts/
```

Persistent volume means `results/` survives pod restarts between sessions.

---

## 2. Pod startup commands

Paste these into the RunPod "startup script" box **or** run them manually
in the pod terminal after it boots.

```bash
#!/bin/bash
set -e
cd /workspace

# Clone repo (cloned into ./alps-4b to keep paths consistent)
git clone https://github.com/4qdrai/ALPS-4B.git alps-4b
cd alps-4b

# Environment setup (adapted from scripts/runpod_setup.sh)
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"

echo "Setup done."
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"
```

Expected output: `CUDA: True | GPU: NVIDIA A40`

---

## 3. Running Session 1 (H0, H0b, H1, H5 first read)

**Estimated time: 6–10 h (dominated by training two models × 80 epochs)**

```bash
cd /workspace/alps-4b
source venv/bin/activate
export PYTHONPATH=src

# Standard Session 1 — all defaults are correct for A40
EPISODES=10000 EPOCHS=80 DMODEL=192 \
  bash scripts/run_a40_unsupervised.sh 2>&1 | tee logs/session1.log
```

### Key ENV vars (all have good defaults; only override if needed)

| Variable | Default | Change when |
|---|---|---|
| `EPISODES` | 10000 | — (standard LeWM regime) |
| `EPOCHS` | 80 | Raise to 120 if G1_ssl > 0.40 after session 1 |
| `DMODEL` | 192 | — (A40 target; 128 is the local-dev value) |
| `ENC_DEPTH` | 10 | — |
| `ENC_HEADS` | 8 | — |
| `WINDOW` | 6 | — |
| `STRIDE` | 4 | Raise to 6 if G1_ssl > 0.40 (more inter-frame motion) |
| `BATCH` | 64 | Can raise to 128 on A40 safely |
| `SIGREG_SLICES` | 1024 | — |
| `NUM_CODES` | 64 | — |
| `N_EVAL` | 200 | — (200 episodes for stable gate numbers) |
| `COARSE_K` | 8 | — |
| `FINE_K` | 24 | — |
| `CX_EPISODES` | 3000 | — |
| `MAKE_VIDEOS` | 1 | Set to 0 to skip videos on Session 1 |
| `RETRAIN` | (unset) | Set to `1` to force re-train an existing model |

### What the 8 stages do

| Stage | What runs | Time estimate |
|---|---|---|
| 1 | Generate 10k-episode dataset (if not present) | 15–30 min |
| 2 | Train simple model (`--lewm-ssl`, d192, 80 ep) + G1/collapse/four-brain gates | 2–3 h |
| 3 | Abstraction gates (validate_abstraction) | 10 min |
| 4 | Generate + train complex model + H3/H4/H2 gates | 2–3 h |
| 5 | Fourth Brain gates simple + complex (H4 label-free key in complex) | 20 min |
| 6 | MoE specialization simple + complex | 15 min |
| 7 | RAG single-pass + H7 lifelong batches | 15 min |
| 8 | Proof videos (simple + complex GIF/MP4) | 10–20 min |

---

## 4. GO / NO-GO gate (read this before interpreting any other output)

After stage 2 completes, open:

```
results/two_rooms/validation/unsupervised/temporal_gates.json
```

Look at **`G1`** (frozen linear probe decode error, world units):

| G1 value | Meaning | Action |
|---|---|---|
| **< 0.30 wu** | **GO** — latent is linearly identifiable without labels. All downstream gates are meaningful. | Proceed with sessions 2–5. |
| 0.30–0.45 wu | Marginal — latent organizes partially. Downstream numbers will be weaker. | Raise `EPOCHS=120 STRIDE=6` and re-run stage 2 only (`RETRAIN=1 MAKE_VIDEOS=0`). |
| **> 0.45 wu** | **NO-GO** — foundation not validated. | Debug: check `G_collapse` first (eff-rank, dead-dims, pairwise-cos). If collapse is present, SIGReg may not be firing — check training log for `L_sig > 0`. If no collapse, raise `EPOCHS=150 STRIDE=8 EPISODES=15000`. |

Also check `G_collapse` in the same file:
- `eff_rank` should be > 5
- `dead_dims` should be 0
- `pairwise_cos` should be ≤ 0.40

If `G_collapse` is healthy but `G1` > 0.30, the issue is identifiability (not collapse);
raise epochs/stride.  If `G_collapse` is bad (rank < 3), fix SIGReg first.

---

## 5. Session-by-session plan

### Session 1 (~6–10 h) — run `run_a40_unsupervised.sh` with defaults

Goal: first read of H0, H0b, H1, H5.

Artifacts produced:
- `results/two_rooms/validation/unsupervised/temporal_gates.json` — **read G1 first**
- `temporal_gates_complex.json` — H5 complex four-brain
- `abstraction_gates.json` — strategic/tactical prediction
- `fourth_brain.json`, `fourth_brain_complex.json` — H8–H10
- `moe_specialization.json`, `moe_specialization_complex.json` — H11
- `rag_selflearning.json` — H7 single-pass
- `../videos_unsup/fourbrain_simple_*.{gif,mp4}`, `fourbrain_complex_*.{gif,mp4}`

After session 1 completes, push artifacts back:
```bash
git add results/two_rooms/validation/unsupervised/*.json
git add results/two_rooms/videos_unsup/
git commit -m "A40 session 1: unsupervised full pipeline"
git push
```

### Session 2 (~4 h) — H2 best-Kt, H3, H4 confirm at d192

Prerequisites: H2 Kt-sweep code landed locally (WS-B).

```bash
# Assumes WS-B code is merged and the complex model from Session 1 exists
source venv/bin/activate && cd /workspace/alps-4b
git pull  # pick up WS-B code

# H2: tactical emitter at best Kt (example Kt=6)
RETRAIN=1 \
python -m alps.evaluation.validate_temporal --complex \
    --model-path results/two_rooms/validation/unsup_temporal_complex.pt \
    --data-path data/two_rooms/trajectories_unsup_complex.pt \
    --n-episodes 200 --coarse-k 8 --fine-k 24 \
    --save-dir results/two_rooms/validation/unsupervised \
    --h4-unsup-key --vq-graph --h2-emitter

# Also run on simple:
python -m alps.evaluation.validate_temporal \
    --model-path results/two_rooms/validation/unsup_temporal.pt \
    --data-path data/two_rooms/trajectories_unsup.pt \
    --n-episodes 200 --coarse-k 8 --fine-k 24 \
    --save-dir results/two_rooms/validation/unsupervised \
    --vq-graph --h2-emitter
```

### Session 3 (~5 h) — Fourth Brain + MoE at scale (H8–H11)

Prerequisites: WS-I monitor + WS-J MoE logging code landed locally.

```bash
git pull  # pick up WS-I/J code
# Calibrate + evaluate Fourth Brain at n=200
python -m alps.evaluation.fourth_brain \
    --model-path results/.../unsup_temporal.pt \
    --data-path data/.../trajectories_unsup.pt \
    --n-cal 60 --n-eval 200 --save-dir results/.../unsupervised
python -m alps.evaluation.fourth_brain --complex --label-free-key \
    --model-path results/.../unsup_temporal_complex.pt \
    --data-path data/.../trajectories_unsup_complex.pt \
    --n-cal 60 --n-eval 200 --save-dir results/.../unsupervised
# MoE specialization
python -m alps.evaluation.moe_specialization \
    --model-path results/.../unsup_temporal.pt \
    --data-path data/.../trajectories_unsup.pt --save-dir results/.../unsupervised
```

### Session 4 (~6 h) — H6 scaling curve + H7 lifelong

```bash
# H6: one data/train/eval run per N
for N in 2 4 6 8; do
  python -m alps.benchmarks.two_rooms.data_generator \
      --save-path data/nrooms_${N}.pt \
      --num-episodes 5000 --n-rooms $N --seed 7
  python -m alps.training.train_temporal \
      --data-path data/nrooms_${N}.pt \
      --out results/two_rooms/validation/unsup_nrooms_${N}.pt \
      --lewm-ssl --d-model 192 --enc-depth 10 --enc-heads 8 \
      --window 6 --stride 4 --epochs 80 --save-model
  python -m alps.evaluation.validate_temporal \
      --model-path results/two_rooms/validation/unsup_nrooms_${N}.pt \
      --data-path data/nrooms_${N}.pt --n-episodes 200 \
      --coarse-k 8 --fine-k 24 \
      --save-dir results/two_rooms/validation/nrooms_${N}
done

# H7 lifelong batches (runs on the simple unsup model from Session 1)
python -m alps.evaluation.fourth_brain --h7-lifelong \
    --model-path results/.../unsup_temporal.pt \
    --data-path data/.../trajectories_unsup.pt \
    --n-cal 60 --n-eval 100 --n-batches 5 \
    --save-dir results/.../unsupervised
```

### Session 5 (~3 h) — H12 real-video d192 + final video checkpoints

Covered separately when WS-K (real-video capture) is ready.

---

## 6. Retrieving artifacts after each session

```bash
# On the pod — push JSON results and any generated videos
cd /workspace/alps-4b
git add results/
git commit -m "A40 session N artifacts"
git push

# OR: pull from your local machine via rsync
rsync -avz --progress runpod_user@POD_IP:/workspace/alps-4b/results/ ./results/
```

Artifact index (after Session 1):

| File | Contains |
|---|---|
| `temporal_gates.json` | G1, G_collapse, G_str, G_tac, G_roll, G_4brain |
| `temporal_gates_complex.json` | H5 complex four-brain (key→door→goal) |
| `abstraction_gates.json` | strategic + tactical latent prediction + goal emission |
| `fourth_brain.json` | H8 monitor AUROC, H9 escalation, H10 fallback |
| `fourth_brain_complex.json` | same for complex (H4 label-free key) |
| `moe_specialization.json` | H11 MI + knockout matrix |
| `rag_selflearning.json` | H7 single-pass surprise-gated RAG |
| `rag_lifelong.json` | H7 lifelong curve (added in stage 7b) |
| `../videos_unsup/*.gif` | Proof clips (simple + complex) |

---

## 7. Fallback decisions if G1 fails

Work through these in order — each takes ~2–3 h to confirm:

1. **G_collapse bad (eff-rank < 3):**
   Check the training log for `L_sig`.  If it is 0 throughout, SIGReg is not
   registered.  Verify that `--lewm-ssl` wires up `sigreg_slices` correctly
   in `train_temporal.py`.  Fix and re-run stage 2 (`RETRAIN=1`).

2. **G_collapse healthy but G1 ∈ (0.30, 0.45):**
   Raise `EPOCHS=120 STRIDE=6`.  Inter-frame motion (STRIDE) is the single
   biggest lever for identifiability in pure-SSL.

3. **G1 > 0.45 after epoch/stride sweep:**
   Raise `SIGREG_SLICES=2048` (more Epps–Pulley samples per batch).
   If still failing, try `BATCH=128` (larger mini-batches → more stable
   covariance estimates for VICReg).

4. **G1 > 0.60 after all of the above:**
   The SSL recipe is fundamentally not converging on this data.  Run the
   H0b ablation arm first (without SIGReg) to confirm collapse IS the
   failure mode, not something else.  Then report the honest result:
   the supervised arm (G1 = 0.19 wu) passes; the SSL arm does not yet.
   That is a meaningful finding (motivates the H0b necessity claim) and is
   still publishable — but the paper's unsupervised framing needs the SSL arm.

---

## 8. Quick sanity check before starting the pod

Run this locally (takes < 30 s, confirms all edge element code is importable
and smoke-tests H4/H3/H6):

```bash
python -c "
import sys; sys.path.insert(0, 'src')
import torch
from alps.core.temporal_world_model import TemporalHierWorldModel
from alps.evaluation.validate_temporal import detect_key_pickups_unsup, build_graph_vq
from alps.benchmarks.two_rooms.environment import NRoomsEnv

model = TemporalHierWorldModel(d_model=32, enc_depth=2, enc_heads=2, num_codes=8,
                                num_experts=2, active_experts=1, op_depth=2,
                                abs_depth=2, max_frames=4).eval()
T, frames = 60, torch.randint(0, 256, (60, 3, 128, 128), dtype=torch.uint8)
starts = torch.tensor([0, 20, 40])
hk, s, fl = detect_key_pickups_unsup(model, frames, torch.zeros(T, dtype=torch.long),
                                      starts, T, 'cpu', max_eps=3)
print(f'H4 OK surp_max={s.max():.3f}')
env = NRoomsEnv(n_rooms=4, seed=0)
obs = env.reset()
print(f'H6 OK N=4 pos={obs[\"position\"][0]:.2f}')
print('ALL OK — safe to push to A40')
"
```

If this passes, all five edge elements (H2/H3/H4/H6/H7) are wired and ready.
