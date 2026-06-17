# ALPS-4B — Project Handoff (2026-06-14)

*Consolidated state so a fresh session can continue without losing progress.*
*Complements the auto-memory at `…/memory/alps4b-validation-state.md` (loaded each session).*

---

## 0. Coordinates
- **Repo:** `https://github.com/4qdrai/ALPS-4B` — branch `main`. Owner/committer: **`sfreedoms2035 <sfreedoms2035@gmail.com>`**.
- **Local working dir:** `H:\Meine Ablage\SayBouBase\raw\Projects\AIFrontTierChallenge\Synthese\FormulatioofEdgeHypotheses&Evidences\Evidences\ALPS-4B`
- **Local GPU:** RTX 4060 (8 GB) — for *cheap hypothesis confirmation*. **A40 pod** (RunPod, 48 GB) — for *real training* (stable numbers). This LOCAL-confirm / A40-train split is an explicit owner directive.
- **Latest commits:** `e234f78` (spatial OOM cap) ← `c377de5` (fourth_brain --spatial + 2 crash fixes) ← `df1ea46` (a40 spatial) ← `dd286c0` (decoded control).
- **No Claude trailer in commits** (owner had me strip `Co-Authored-By: Claude` from history + re-point a stale tag; force-pushed clean). Do **not** add it going forward.

## 1. The goal (owner's words, the north star)
Show the **clear, disruptive superiority of the 4B (Four-Brain) multi-abstraction-layer architecture**:
- **Unsupervised**, world-model-**prediction**-based (no labels; **SIGReg** stabilization — pure SSL, **no EMA, no stop-grad, no VICReg, no anchored arm**).
- **Strategic** layer reasons in abstract space to achieve strategic goals and **emits strategic constraints/goals for lower layers**; same for **tactical**.
- **Latent-RAG** for long-term experience & learning (adapt to new situations, no weight updates).
- **Self-monitoring loop → escalation → fallback** (the Fourth Brain).
- Ultimately on **real video** to understand the real world, **physics**, and **consequences of agent actions**.
- **Decoding based on the trained predictors** (the model's own op/tac/str predictors + frozen linear read-out), not hand-coded heuristics or supervised anchors.

Toy testbed = **Two-Rooms** (128×128, an agent/dot navigates between rooms through a door; "complex" mode = key→door→goal). It is a *stepping stone* to real video, **not** the end goal. (LeWM's own paper reports Two-Room as its *weakest* env — a tiny agent in a static scene is a misleading regime; keep that in mind.)

## 2. Architecture (what exists, works)
`TemporalHierWorldModel` (`src/alps/core/temporal_world_model.py`): d_model=192, enc_depth=10, enc_heads=8, **~4.88M encoder** (ViT-Tiny class), patch_size=(2,16,16) → **8×8 = 64 spatial tokens**, window W=6, stride S=4.
- **operative** = `CausalTemporalPredictor` 1-step latent prediction (+ action-conditioned AdaLN).
- **tactical** = MoE, predicts K_tac-ahead latent + goal-conditioned sub-goal head.
- **strategic** = VQ (64 codes) concept, predicts K_str-ahead concept.
- **fourth brain** (`evaluation/fourth_brain.py`) = monitors m1(surprise)/m2(off-manifold)/m3(stall) → escalation tiers → fallback to safe landmark.
- **LeWM-SSL** (`--lewm-ssl`): SIGReg on embeddings AND predictions at all 3 scales (BatchNorm pred-projectors), no EMA/stop-grad/VICReg, pos/dyn weights = 0. Verified byte-faithful to LeJEPA.
- **Eval = frozen post-decode**: encoder eval()/no_grad; a fresh linear probe reads position from the frozen latent (standard JEPA protocol); control acts in latent space via the trained predictors.

## 3. THE central finding of this session (the crux — read this)
**Pure-SSL position-identifiability fails through the GLOBAL POOL, and the fix is a SPATIAL READOUT.**

1. Pure-SSL **G1 (pooled position decode) = ~3.25–3.6 wu = random** on Two-Rooms (both local and A40). **No collapse** (SIGReg healthy: eff-rank ~5, dead-dims 0, cos ~0.03). So the latent is non-degenerate but the *pooled* readout is position-blind → the whole hierarchy reads 0.
2. **Diagnosis** (`evaluation/diagnose_g1.py`): position **is** in the spatial tokens — the **full token grid decodes position at R² 0.97**, but the **mean-pool / [CLS] global readout discards it**. The agent is ~2 px of 128² (0.02%); mean-pool dilutes it; a learned [CLS] *learns to ignore* it (op-loss crashes to 0.02 by excluding the agent); inverse-dynamics auxiliary **did not bootstrap** (noisy stride-4 action, tiny controllable signal). These were all tried and **ruled out** as the fix.
3. **THE FIX = spatial/object readout** (`model.spatial_readout(z, grid)`): a coarse g×g average-pool of the tokens, which **preserves the agent's position**. Validated locally on the existing pure-SSL model **with no retraining**:
   | readout | test err | R² |
   |---|---|---|
   | POOLED (global) | 3.6 | 0.02 (random) |
   | SPATIAL 2×2 | 1.80 | 0.75 |
   | SPATIAL 4×4 | 0.99 | 0.92 |
   | SPATIAL 8×8 (full grid) | 0.55 | 0.97 |
   | raw pixels (control) | 0.04 | 1.00 |
4. **Wired `--spatial`** through `validate_temporal` (four-brain simple + complex + key, H2/H3/H4) and `fourth_brain` (monitors + escalation + fallback + RAG), using **predictor-based decoded control**: the trained op predictor → next latent grid → `spatial_readout` → frozen **ridge** decode → predicted agent **position** (action-sensitive) → steer toward the graph waypoint. RAG memory stays d_model (the LatentRAG corrects the operative latent); only the surprise m1 uses the readout.

## 4. THE open blocker (what's NOT yet working)
**The closed-loop hierarchy does NOT route yet** — and we know why:
- The spatial readout decodes position at **~0.55–0.58 wu** (local 0.55, **A40 grid-8 G1_spatial = 0.584**). Crucially, the **A40 (bigger encoder) did NOT sharpen it vs local** → this is a **readout-resolution ceiling** (patch=16 → 8×8 grid, each cell 1.25 wu), **not** a model-capacity limit. More epochs/data won't help.
- Per-step agent motion ≈ **0.27 wu**. Since decode noise (0.58) **> motion (0.27)**, the predictor-decoded control can't reliably pick the right action → **`G_4brain` ≈ 0 expected** (matches every local run; no local model routed cross-room in either pooled or spatial mode).
- **We have NOT yet cleanly demonstrated that the control routes at ANY decode precision** — the one local "sharp" model test was inconclusive (that model didn't route in *pooled* mode either). So the open question is: *is the bottleneck decode precision, or something deeper in the control (stride/action-sensitivity)?*

## 5. Current A40 run state (as of handoff)
- Stage 1 (data, two-pass): **OK**, 675k frames / 33 GB, no OOM.
- Stage 2 training: **DONE**, model saved at `results/two_rooms/validation/unsup_temporal.pt`. Loss plateaued `op ≈ 0.91` (expected weak-dynamics), no collapse (`sig 2.3→0.97`, VQ diversifying).
- Stage 2 validation: printed **`G1_spatial 0.584`** then **OOM-killed** (spatial gather 26 GB at grid 8). **FIXED** (commit `e234f78`: cap ridge-fit probe to 20k/5k). The full pipeline script aborted at that point (`set -e`).
- **We have NOT yet seen `G_4brain`** (the edge). That is the immediate next read.

## 6. NEXT STEPS (in order)
1. **IMMEDIATE — see `G_4brain` cheaply** (the simple model is saved; no retrain): on the pod, `git pull` then
   ```bash
   python -m alps.evaluation.validate_temporal \
     --model-path results/two_rooms/validation/unsup_temporal.pt \
     --data-path  data/two_rooms/trajectories_unsup.pt \
     --n-episodes 200 --coarse-k 8 --fine-k 24 --spatial --spatial-grid 8 \
     --save-dir results/two_rooms/validation/unsupervised
   ```
   Read `G_4brain` (operative vs strategic vs tactical) + `G_collapse`.
2. **Branch on `G_4brain`:**
   - **If it routes** (operative ≈ 0 ≪ graph) → resume the full pipeline: `EPISODES=10000 EPOCHS=80 DMODEL=192 SPATIAL_GRID=8 bash scripts/run_a40_unsupervised.sh` (skips the saved simple model, trains complex, runs all spatial-wired gates: four-brain simple+complex+key, abstraction, fourth-brain H8/H9/H10, RAG H7, MoE, videos).
   - **If it's ≈ 0** (likely) → **decode is resolution-capped above what control needs.** Real fix = **finer patches**: `patch_size (2,16,16) → (2,8,8)` ⇒ 16×16 = 256 tokens ⇒ ~2× sharper decode (~0.3 wu). Needs a small **`--patch-size` CLI flag** added to `train_temporal.py` + threaded to the model, then **retrain** (~8 h simple).
     - **CAVEAT:** ~0.3 decode ≈ 0.27 motion = still borderline. May *also* need a **larger-displacement control** (predict/act over multiple steps, e.g. use the stride-4 / tactical K-ahead prediction so the per-decision displacement (~0.73 wu) exceeds the decode noise). Consider testing this control change *before* the patch retrain, since it might rescue routing at the *current* 0.58 decode.
3. **After routing works:** confirm the rest of the edge at scale (all already spatial-wired): strategic/tactical **goal emission** (`validate_abstraction` + the four-brain graph routing), **latent-RAG H7** (lifelong, zero weight updates), **self-monitoring H8 / escalation H9 / fallback H10**, MoE H11. These are the owner's priority list.
4. **Then real video (H12):** the spatial/object readout + **latent actions** (inverse-dynamics works at scale per CLAW/AdaWorld 2026 — the toy's tiny-agent pathology won't exist when the controllable content is a large fraction of the frame). Full design + 2026 literature in **`docs/RESEARCH_4B_SCALEUP.md`**.

## 7. Key files
- `src/alps/core/temporal_world_model.py` — model; `spatial_readout`, `pool`/`tok_pool`, `inverse_action`, `inv_head`, `use_cls_pool`.
- `src/alps/core/encoders.py` — `VisionEncoder` (has `use_cls_token`); **`patch_size` is a constructor arg but NOT yet a CLI flag** (this is what the "finer patches" step needs).
- `src/alps/training/train_temporal.py` — `--lewm-ssl`, `--cls-pool` (off), `--inv-dyn`/`--inv-weight` (off), all the loss terms. **Add `--patch-size` here.**
- `src/alps/evaluation/validate_temporal.py` — G1/collapse/four-brain; **`--spatial`** path: `fit_ridge_decode`, `run_episode_spatial`, `gate_four_brain_spatial`, `build_graph_raw(readout=…)`; `detect_key_pickups_unsup`/`build_graph_vq`/`gate_tactical_emitter` (H4/H3/H2).
- `src/alps/evaluation/diagnose_g1.py` — the readout probe (pooled vs spatial 2×2/4×4 vs full grid vs pixels + inter-frame motion). Run this to re-confirm the readout finding on any model.
- `src/alps/evaluation/fourth_brain.py` — monitors/escalation/fallback/RAG, fully `--spatial`-wired.
- `scripts/run_a40_unsupervised.sh` — 8-stage pipeline; env vars `EPISODES/EPOCHS/DMODEL/STRIDE/SPATIAL_GRID/N_EVAL/…`.
- Docs: `A40_TRAINING_INSTRUCTIONS.md`, `RESEARCH_4B_SCALEUP.md`, `EDGE_PROGRAM.md`.

## 8. Bugs found & fixed this session (so they don't resurface)
- **Data-gen OOM**: `np.stack(list_of_frames)` held ~2× the 33 GB dataset → cgroup-killed. Fixed with **two-pass generation** (`data_generator.py`, `generate_complex.py`). (A naive "free-as-you-fill" did NOT work: 48 KB frames are below glibc's mmap threshold so frees don't return RSS.)
- **`repr_decoder_gate.py`**: top-level `import matplotlib` crashed stage-2 on pods without it → made **lazy** (inside `make_figures`).
- **`fourth_brain.py` two pre-existing crashes** (would have killed A40 stages 5 & 7): (a) `run_episode_fb` call passed 2 stray args → `thr` double-bound → `TypeError`; (b) `plan_waypoints` was **used but never defined** → `NameError`. Both fixed.
- **Spatial probe OOM** at grid 8 (26 GB gather) → capped to 20k/5k samples.

## 9. Process gotchas (learned the hard way)
- **Piping a run through `| grep` re-buffers output** even with `PYTHONUNBUFFERED=1` → the log looks empty until the process exits. For live monitoring: `PYTHONUNBUFFERED=1 python … ` with **no pipe**.
- A **machine going to sleep silently kills long local runs** (orphans the process, discards buffered output). Keep the box awake.
- A40 pods may show huge host RAM via `free -h` but be **cgroup-capped** lower — watch for `Killed` even when `free` looks fine.
- Local Two-Rooms is **noise-limited**; no local model routes cross-room cleanly. Stable closed-loop numbers need the A40.

## 10. One-line status
*Representation under pure SSL is solved via the spatial readout (position recoverable, G1_spatial 0.584 vs random 3.6). The closed-loop hierarchy edge is still open — decode precision (0.58) is resolution-capped above the per-step motion (0.27), so control likely won't route until we either (a) sharpen the decode with finer patches, and/or (b) use a larger-displacement control. The immediate next action is to read `G_4brain` on the saved A40 model (cheap, model already trained) to decide which.*
