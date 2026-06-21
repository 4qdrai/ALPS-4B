# ALPS-4B — Project Handoff (2026-06-14)

*Consolidated state so a fresh session can continue without losing progress.*
*Complements the auto-memory at `…/memory/alps4b-validation-state.md` (loaded each session).*

---

## ⭐ REFINED MASTER PLAN (2026-06-21) — supersedes §4–§6 and §9b below

### Where we actually are (Run-2 = FIRST real closed-loop measurement)
Fresh A40 patch-16 model `unsup_temporal.pt` (d192/depth10/W6/S4, `--lewm-ssl`, 80 ep; healthy: op 0.786, sig 0.91, no collapse, enc 4.88M), **backed up to HF `Free2035/alps-4b-tworooms`** (model-loss-proof after the pod was deleted twice). On a 1500-ep `_eval_small.pt` (N=103,113):
- **G1_spatial = 0.730 wu** (global-pool 3.518) → spatial readout recovers position ~4.8×. Representation edge holds.
- **G_4brain = 0** at `--ctrl-k 2`: operative 0.00 → +strategic 0.00 → +tactical 0.00, **oracle 0.97**, necessity +0.00/+0.00.

**THE REFRAME — this collapses to ONE bottleneck.** Oracle 0.97 proves the task + k-means graph + waypoints + planner are all correct (with TRUE position the hierarchy solves cross-room 97%). The only broken link is **execution**: decoded position (0.730 wu) is noisier than one step of motion (0.27 wu), so decoded control can't pick the right action → all tiers read 0 and become incomparable. **Planning is almost certainly already correct; execution decode-precision is the wall.** The whole plan is: get decoded execution below the motion scale, then bank each edge.

### The claim ladder (what "all edges proven" means — acceptance criteria)
| # | Edge | Proven by | Target | Status |
|---|---|---|---|---|
| E1 | Unsup representation | G1_spatial ≪ random | ≫ 3.5; ↓ with patch8 | ✅ 0.73 vs 3.5 |
| E2 | **Hierarchical PLANNING** | tier plan @ fixed exec: op ≪ str ≤ tac (cross-room) | tac ≥ 2× op | ⏳ Track B (decode-independent) |
| E3 | Unsup closed-loop SIMPLE | decoded control solves; op ≪ tac | tac ≥ 2× op, → oracle | ⏳ gated on decode |
| E4 | Unsup closed-loop COMPLEX | routes start→key→goal; op=0 ≪ tac | tac ≥ 0.5·oracle | ⏳ gated on decode < 0.55 pickup |
| E5 | Latent-RAG H7 | lifelong, 0 weight updates | gain ≥0.10, interf ≤0.02 | ⏳ needs a routing model |
| E6 | Fourth Brain H8/9/10 | monitor AUROC, escalation lift, safe fallback | AUROC ≥0.8, safe-reach ≥0.8 | ⏳ needs a routing model |
| E7 | MoE H11 | expert↔regime MI + knockout | diag-dominant | ✅ passes locally |
| E8 | **Proof videos** SIMPLE+COMPLEX | side-by-side op-stalls vs 4B-solves | both solve on screen | ⛔ blocked by **V1** |

### Two tracks
**Track A — full unsupervised closed-loop (north star):** sharpen decoded execution until it routes, then run the whole suite + videos.
**Track B — decode-independent planning proof (de-risk E2 NOW):** execute each tier's UNSUPERVISED plan with a fixed near-oracle controller → proves "hierarchy > flat" even while decode sharpening is in flight.

### Track A — sequenced, with the decision tree
- **A1 (running now): ctrl-k sweep, NO retrain.** `--ctrl-k 5` on the current model (5×0.27 = 1.35 > 0.730 → SNR ~1.8). If 0, try `--ctrl-k 8`. Branch on the result:
  - operative > 0 **and** tac > op → **routing achieved at patch16** → jump to A4 + A5.
  - operative > 0 **but** tac ≈ op → execution works, hierarchy not separating → run Track B + inspect graph/waypoints.
  - still 0 → decode too coarse even amplified → **A2**.
- **A2: sharpen decode — retrain at patch8 (the real fix).** Now one command (script is patch-tunable as of today): `PATCH="2 8 8" SPATIAL_GRID=16 CTRL_K=3 RETRAIN=1 EPISODES=10000 EPOCHS=80 bash scripts/run_a40_unsupervised.sh`. 16×16 = 256 tokens → decode ~0.3 wu. **Back up both models to HF before anything else.** Re-read G1_spatial (~0.3) + G_4brain.
- **A3: combine if borderline.** ~0.3 decode ≈ 0.27 motion → pair patch8 with `CTRL_K=3..5` (the script now threads ctrl-k into every gate).
- **A4: full edge suite (once op>0, tac>op).** Same script runs it all on the routing model: four-brain simple+complex+key (E3/E4), abstraction goal-emission (E2 @ scale), fourth-brain H8/9/10 (E6), RAG H7 (E5), MoE H11 (E7).
- **A5: complex closed-loop (E4).** patch8 complex model (script trains it) → decode < 0.55 pickup radius so the agent can physically touch the key; operative 0 (can't detour for the key) ≪ tactical (routes start→key→door→goal). The most dramatic hierarchy demo.
- **A6: proof videos (E8)** — runs only after **V1**.

- **V1 (REQUIRED code, found 2026-06-21): spatial-wire `make_videos_4b.py`.** It currently drives the Four-Brain panel on the **global pool** (`model.pool`, `hist_greedy_action_latent`, `build_graph_raw` over pooled `fit_probe`) = the position-blind path we proved fails → the videos would film the Four-Brain **stalling** even when the gates route. Fix = mirror `validate_temporal.run_episode_spatial`: build the graph on `spatial_readout`, decode waypoints + live position via the spatial **ridge** decode, drive control with `buf.rollout_decode(decode_state, a, ctrl_k)`, add `--spatial/--spatial-grid/--ctrl-k` flags, pass them from script stage 8. Smoke-renderable on any model; becomes the proof the moment a model routes. **Implement while the pod trains.**

### Track B — decode-independent planning proof (do this to bank E2)
Add a `true_pos_exec` flag to the four-brain gate: keep each tier's **unsupervised** plan (operative = no plan; strategic = coarse graph; tactical = fine graph — all built on frozen SSL latents), but execute the chosen waypoint with the env's true-position controller. Removes decode precision as a confound → directly measures the PLANNING edge (expect op ≪ str ≤ tac). Honest framing: "the unsupervised hierarchy PLANS the route; execution shown separately." Small addition to `run_episode_spatial`.

### Open code gaps to verify (so E2/E6/E7 aren't measured through a position-blind decoder)
- `validate_abstraction` goal-emission gates (G_str2tac / G_tac2op) decode subgoal POSITIONS via `tac_pos_head` (pooled) → position-blind under pure SSL. Verify they read the spatial readout, else their "points toward goal" check is meaningless on the `--lewm-ssl` model.
- `moe_specialization` knockout Δerr is on the pooled tac-decode — fine for relative routing/MI, but note the absolute decode is pooled.

### Discipline (lessons from losing the pod twice)
- **Back up every trained model to HF immediately** (`HfApi.upload_file`, `$HF_TOKEN` from env; never echo the token; revoke the exposed one).
- `--n-episodes 30` for fast reads; the silent control loop is NOT a hang (progress prints since `84917be`). Keep the box awake; don't pipe runs through `grep`.

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

## 5. Current A40 run state (UPDATED — pod was STOPPED by owner)
- Stage 1 (data, two-pass): **OK**, 675k frames / 33 GB, no OOM.
- Stage 2 training: **DONE**, model saved at `results/two_rooms/validation/unsup_temporal.pt`. Loss plateaued `op ≈ 0.91` (expected weak-dynamics), no collapse (`sig 2.3→0.97`, VQ diversifying).
- **`G1_spatial` CONFIRMED at A40 scale = 0.544–0.584** (grid 8; pooled G1 ~3.5). Same as local → readout-resolution floor, *not* model-limited. Representation claim holds; closed-loop still the open question.
- Two OOMs hit and fixed: spatial ridge-probe gather (commit `e234f78`, cap 20k/5k); then validation on the **full 33 GB frame tensor + grid-8 ops** stayed too tight on the cgroup → **WORKAROUND: validate on a small eval subset** (`python -m alps.benchmarks.two_rooms.data_generator --save-path data/two_rooms/_eval_small.pt --num-episodes 1500 --max-steps 100 --heuristic-fraction 0.4 --seed 7`, ~5 GB). That cleared the OOM and printed `G1_spatial 0.544`.
- **"Stuck" was a FALSE ALARM** — not a hang: after `G1_spatial` the spatial four-brain runs **`n-episodes 200 × 3 tiers × ~140 steps × 4 predictions` ≈ 336k predictor passes + two 12288-d k-means, ALL SILENT** (~20–40 min) until `G_4brain` prints. Owner read the silence as stuck and **stopped the pod**. FIXED: added **progress prints** (commit `84917be`) so each tier + the graph build now report. **For a fast read, use `--n-episodes 30`** (~5 min) not 200.
- **We have STILL NOT seen `G_4brain`** (the edge). That is THE immediate next read on the next pod.

## 6. NEXT STEPS (in order)
1. **IMMEDIATE — see `G_4brain` cheaply** (model saved, no retrain). On a fresh pod, clone, `pip install -e .`, then make a SMALL eval set (avoids the 33 GB OOM) and run with **few episodes + progress prints** so it's fast and visibly not-hung:
   ```bash
   python -m alps.benchmarks.two_rooms.data_generator --save-path data/two_rooms/_eval_small.pt \
     --num-episodes 1500 --max-steps 100 --heuristic-fraction 0.4 --seed 7
   python -m alps.evaluation.validate_temporal \
     --model-path results/two_rooms/validation/unsup_temporal.pt \
     --data-path  data/two_rooms/_eval_small.pt \
     --n-episodes 30 --coarse-k 8 --fine-k 24 --spatial --spatial-grid 8 \
     --save-dir results/two_rooms/validation/unsupervised
   ```
   (The trained model `unsup_temporal.pt` lives on the stopped pod's volume — if that volume is gone, the ~8 h stage-2 training must be redone via `bash scripts/run_a40_unsupervised.sh` first.) Read `G_4brain` (operative vs strategic vs tactical) + `G_collapse`.
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

## 9b. LATEST (both decode levers now in code; edge still unmeasured)
- **Two committed levers for `decode(0.55) > motion(0.27)`:** `--ctrl-k K` (K-step rollout lookahead, **no retrain**, `99ec10a`/`64afe43`) raises displacement above the decode noise; `--patch-size 2 8 8` + `--spatial-grid 16` (**retrain**, `99ec10a`) sharpens the decode. Grid can't be finer than the token grid (at patch16, grid8 = full 64-token grid = finest; finer grid only with patch8 → 256 tokens → grid16).
- **`--ctrl-k` is ~K× slower** (K predictor passes/action). A local test on the depth-6 pure-SSL model **timed out (400 s) + inconclusive** (operative/strategic 0.0 partial, but that's an undertrained model with a drifty rollout — NOT a verdict). `trajectories_large.pt` corrupted locally.
- **`G_4brain` is STILL UNMEASURED.** The definitive test: on the A40, run `validate_temporal --spatial --spatial-grid 8 --ctrl-k 3 --n-episodes 30` on `unsup_temporal.pt` (better predictor → fair rollout). Try `--ctrl-k 2` first (lighter); bump to 5 if needed.

## 10. One-line status
*See the ⭐ REFINED MASTER PLAN at the top — it supersedes this and §4–§6/§9b. In short (2026-06-21): `G_4brain` is now MEASURED for the first time = 0 at ctrl-k 2, with **oracle 0.97** → the task/graph/planner are all correct; the sole wall is execution decode-precision (G1_spatial 0.730 > motion 0.27). Path: ctrl-k sweep (no retrain, in flight) → patch8 retrain (decode ~0.3) → full edge suite + complex → videos (blocked on V1: `make_videos_4b` is still on the position-blind global pool and must be spatial-wired). Track B (tier-plan @ true-position exec) banks the planning edge E2 independent of the decode wall.*
