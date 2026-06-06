# ALPS-4B — Validation Plan & Falsifiable Evidence Program

This document is the honest, runnable plan to *prove the edge* of the ALPS-4B
(Four-Brain) architecture on the Two Rooms benchmark (simple + complex), to
visualize the benefit of the tactical/strategic layers, and to validate the
planning, decoder, self-learning, and latent-graph ideas. Every claim here is
tied to a script and an acceptance gate, so results are reproducible and the
program fails fast when a component does not work.

> TL;DR of the starting state (measured, not asserted): the shipped
> `two_rooms_model.pt` scores **5% overall / 0% cross-room** on navigation, its
> latent decodes position to **3.26 world units** (≈ random in a 10-unit room),
> and its predictor is **action-insensitive** (sensitivity/error ratio 0.02).
> The "0.002 units decoding error" in the old report was the *latent prediction
> MSE* mislabeled as a position error. The architecture concept is sound; the
> formulation needed fixing, not more compute.

---

## 0. Why the old result failed (root causes)

| # | Root cause | Evidence (file) |
|---|---|---|
| 1 | Degenerate temporal input: a single frame is broadcast to T identical frames; agent moves ~0.3 units (~4 px). Predictor minimizes MSE by predicting near-identity. | `train_two_rooms.py:encode_single_frame`, `frame_skip` default 1 |
| 2 | Position aux loss OFF by default → latent never forced to encode position. | `train_two_rooms.py` `pos_loss_weight=0.0` |
| 3 | Action conditioning near-zero (zero-init AdaLN, global modulation). | `predictor.py` AdaLN |
| 4 | Planner scores L2 in a latent that barely moves → no signal. Hierarchy never used in control (CEM only calls the operative predictor); cross-room "plan" is a hard-coded door image. | `planner.py:_render_door_obs`, `CEMPlanner.plan` |
| 5 | Decoder evidence mislabeled (latent MSE printed as world-unit error). | `evaluate_two_rooms.py:1236` |
| 6 | Complex mode never trained or evaluated; self-learning/latent-graph never validated; no graph object exists. | (absence of `*complex*` results) |

---

## 1. Acceptance gates (run these in order; do not skip ahead)

Each gate has a hard threshold. A failed gate blocks the next stage.

### G1 — Decoder gate (representation is position-decodable)
Freeze the encoder, train an **independent** probe, report **held-out mean
Euclidean error in world units**.
- **PASS: < 0.3 world units.**
- Script: `python -m alps.evaluation.repr_decoder_gate train --frame-skip 4 --pos-weight 1.0 --save-model`

### G2 — World-model gate (actions move the latent, decodably)
- `action_sensitivity = mean_{i≠j} ||pred(z,a_i) - pred(z,a_j)||`
- `one_step_error = ||pred(z,a_true) - z_next||`
- **PASS: ratio = sensitivity / error > 2.0**, AND dynamics-decode ≈ static-decode ≪ step displacement, AND directional consistency ≥ 0.75.
- Same script (runs G1 and G2 together).
- Baseline-on-shipped-model: `python -m alps.evaluation.repr_decoder_gate probe-existing --ckpt results/two_rooms/two_rooms_model.pt` (the "before").

### G3 — Planning ablation ladder (the edge)
Run identical navigation eval across rungs; the edge holds only if higher rungs
beat lower ones on **cross-room** success at reasonable compute.
- Script: `python -m alps.benchmarks.two_rooms.run_ablation_ladder --model-path results/two_rooms/validation/repr_world_model_fs4.pt --n-episodes 200`
- Rungs:
  - `rung0_random` — floor
  - `rung2_operative_greedy` — world model + decoder, goal-only MPC (expected to stall at the wall on cross-room → the local minimum)
  - `rung4a_strategic_doorgoal` — hand-specified [door, goal] waypoints
  - `rung4b_latent_graph` — waypoints from the learned latent transition graph shortest path **(the real strategic layer)**
  - `rung5_oracle` — near-optimal on true state (ceiling)
- **PASS: rung4b cross-room ≥ 2× rung2 cross-room**, and overall ≥ 80%.
- Report: success/SPL bar chart + `latent_graph.png`.

### G4 — Complex mode (key-gated 4-room)
Same ladder with `--complex`; a flat reactive controller cannot represent
"fetch key → unlock door → goal", so the strategic/latent-graph rung should win
decisively.
- **PASS: full model keyed cross-room ≥ 60%, ≫ flat baseline.**

### G5 — Self-learning (Latent-RAG), validated honestly
- Script: `python -m alps.evaluation.self_learning_validation`
- Splits: WRITE (one-shot recall), TEST (generalization to unseen similar
  contexts), CONTROL (interference). 
- **PASS: WRITE reduction > 50%, TEST reduction > 5%, CONTROL change > −5%.**
- Then demonstrate sleep-consolidation distills RAG → weights with cache purged
  and performance retained.

---

## 2. Modules implemented for this plan

| Module | Purpose |
|---|---|
| `src/alps/evaluation/repr_decoder_gate.py` | Fixes 1–3 + gates G1/G2; `train` and `probe-existing` modes. Defines `ReprWorldModel` (rung 1/2). |
| `src/alps/benchmarks/two_rooms/world_model_planning.py` | Position-space CEM-MPC, wall penalty, oracle/random baselines, SPL, eval harness. |
| `src/alps/core/latent_graph.py` | The missing latent transition graph: cluster latents → landmarks, accumulate transition edges, Dijkstra shortest-path → decoded waypoints. `add_transition` for online self-learning. |
| `src/alps/benchmarks/two_rooms/run_ablation_ladder.py` | Runs G3/G4, saves the hierarchy-benefit chart and graph figure. |
| `src/alps/evaluation/self_learning_validation.py` | G5 with WRITE/TEST/CONTROL splits. |

---

## 3. A40 (48 GB) execution plan

The task is tiny in FLOPs; correctness is the bottleneck. Pre-encode the dataset
once with the frozen encoder so the planning rungs are fast.

1. **Regenerate larger data** (the shipped set is only ~4.1k frames / 100 eps):
   `python -m alps.benchmarks.two_rooms.data_generator --num-episodes 5000 --max-steps 100`
   and `--complex-mode` for the 4-room variant.
2. **G1/G2** — train ReprWorldModel (`--frame-skip 4 --pos-weight 1.0`, a few hundred epochs; `--sigreg-slices 256` keeps it light). **Gate.**
3. **G3** — ablation ladder, `--n-episodes 200`. **Gate.**
4. **G4** — complex mode. **Gate.**
5. **G5** — self-learning + sleep consolidation.

Tuning knobs if a gate fails:
- G1 fails → raise `--pos-weight`, train encoder longer, increase data.
- G2 fails → strengthen action conditioning (larger action embedding / per-patch
  FiLM with non-zero init / inject action as a token), and ensure `frame-skip`
  makes displacement visible.
- G3 cross-room flat≈graph → the graph isn't capturing room connectivity; raise
  `--graph-k`, check `latent_graph.png` edges cross only through the door region.

---

## 4. Claims hygiene

Scope the README/paper to what the ladder demonstrates. Replace unbacked
superlatives ("42 advantages", "comprehensively solving", "mathematically proven
safe", "zero-shot", "O(1)", ">98% error reduction", "instant fleet learning")
with the measured statement, e.g.: *"the latent-graph strategic layer lifts
cross-room success from X% (operative-only) to Y% at Z replans/episode, vs. an
oracle ceiling of W%."* A trustworthy small claim beats 42 unbacked ones.
