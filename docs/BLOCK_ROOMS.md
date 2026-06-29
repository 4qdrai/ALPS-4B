# Block-Rooms — a consequence-dominant, fully-observed testbed for the predictor

## Why
Two-Rooms (a ~2px agent navigating a large static room) is the wrong testbed for the
imagine-and-act thesis, established mechanistically over ~15 experiments:
- **Full view** → the controllable signal is ~2% of a static scene; the SSL predictor aces the
  boilerplate and gets the controllable read-out *actively wrong* (measured: 1-step latent
  imagination error = **1.10 relative**, i.e. worse than predicting nothing).
- **Limited/egocentric view** → new unobserved content enters the frame as the agent moves →
  unpredictable by construction (imagination error **1.86**).

These are independent and fundamental to *navigation in a large space*. The hierarchy needs space;
the predictor needs the action's consequence to be **observed and dominant**. In a simple navigation
toy those are mutually exclusive — which is exactly why LeWM reports TwoRoom as its weakest env
(it works on PushT / OGBench-Cube, which ARE consequence-dominant manipulation).

## What
A **large textured-free block** (the agent) is moved by a **large, deterministic step** in an
**open, fully-observed** arena. Each action produces a big (~14% of pixels), fully-observed,
action-determined change → the SSL predictor MUST learn the dynamics.

`TwoRoomsEnv(block_mode=True)` (in `environment.py`):
- `BLOCK_RENDER_RADIUS = 1.7` (~9% of the frame), `BLOCK_STEP_SCALE = 7.0` (step = 2.1 wu/action).
- Top-down, fully observed (no egocentric, no walls in the minimal mode). Position trivially
  decodable from the block's location (~19% frame diff between far positions).
- Measured: one-step frame change ~14% (vs ~1% for the dot); deterministic.

Threaded through `data_generator.py --block-mode` and `evaluation/diagnose_control.py --block-mode`.

## Validation ladder
1. **`imag relative`** (predictor 1-step latent imagination) must fall from **1.10 → < 0.3**.
   This is the make-or-break number; a ~25-min quick model gives it.
2. **`direction_acc`** (decoded / latent-space / inverse) must clear **0.6** — the predictor's
   imagination driving control, no crutches. (Open arena = basic goal-seeking; the barrier/gate
   for the hierarchy edge is added once the predictor works.)
3. Then add the **wall+gap (simple) and switch-gate (complex)** for the routing edge, run the full
   four-brain pipeline (graph waypoints as strategic/tactical goals + predictor-imagination control),
   and render the autonomous solving videos.

## Unchanged
Pure SIGReg SSL (`--lewm-ssl`), abstraction layers, inverse dynamics, latent graph, RAG,
self-monitor, all gates, all CLI. Only the environment the agent acts in changes.

## Quick validation command
```
PYTHONPATH=src python -m alps.benchmarks.two_rooms.data_generator --save-path data/two_rooms/_block.pt --num-episodes 800 --max-steps 60 --heuristic-fraction 0.4 --seed 5 --block-mode
PYTHONPATH=src python -m alps.training.train_temporal --data-path data/two_rooms/_block.pt --out results/two_rooms/validation/_block.pt --lewm-ssl --inv-dyn --inv-weight 1.0 --d-model 192 --enc-depth 8 --enc-heads 8 --window 6 --stride 1 --batch-size 64 --sigreg-slices 512 --num-codes 64 --num-experts 4 --active-experts 2 --epochs 22 --patch-size 2 16 16 --save-every 5 --save-model
PYTHONPATH=src python -m alps.evaluation.diagnose_control --model-path results/two_rooms/validation/_block.pt --data-path data/two_rooms/_block.pt --spatial-grid 8 --n-steps 200 --block-mode
```
