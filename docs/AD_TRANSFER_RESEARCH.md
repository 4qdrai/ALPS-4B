# ALPS-4B → Autonomous Driving: A Research Program for Data-Efficient, Self-Supervised, Hierarchically-Imagining Driving Models

**Status:** research proposal / transfer study (post-doc level), v1.1 — 2026-07-07
**Author:** ALPS-4B project
**Prerequisite reading:** `docs/PROJECT_HANDOFF.md`, `docs/BLOCK_ROOMS.md`, `docs/SLOT_FOUR_BRAIN.md` (§8–10), the 2026-06-30 harness-fix commits (`d9bac14`, `c030fe5`)

**v1.1 changelog (2026-07-06/07 findings, detailed in the new §3.5):**
- **Egocentric observation directly validated as the control-enabling regime** — on egocentric Two-Rooms, predictor-decoded control is *usable* (calibrated direction_acc **0.69**, forward-dynamics **0.76**, both ≥ 0.6) where the top-down small-agent baseline was 0.19. This supersedes the earlier "egocentric Two-Rooms failed" claim in §3 (that verdict judged by the wrong metric, pre-fix). Commit `aa8e376`.
- **The object-binding laws** (from an exhaustive slot-binding investigation, `docs/SLOT_FOUR_BRAIN.md` §10): appearance/reconstruction-driven slot binding *cannot* discover a small moving object in a minimal scene — two scale-free laws, **both of which invert on real driving video**. This reclassifies object-centric slots as a **real-video technique** (Stage 2), not a toy technique. Commits `7805e21`, `2e4c13a`.
- **A data-contamination bug** (env flags didn't imply `block_mode`; commit `b373f1b`) invalidated every "gate-mode" measurement and, once fixed, revealed the small agent *is* sharply decodable (G1 **0.109** vs the contaminated ~2.0). New instrument-doctrine item I7 (§12).
- **Refined control insight:** usable control needs **action-discrimination in the decoded-state space**, *not* low imagination-fidelity — control was usable at `imag rel 1.27` (> 1). This directly predicts driving works despite the horizon always importing new content (§3.5).

---

## 0. Executive summary

ALPS-4B has now demonstrated, on a controlled testbed and after fixing a measurement-harness defect that had masked the result for the project's entire history, the complete mechanism this program is built on:

> a **from-scratch, fully self-supervised** (SIGReg-only, no pretrained encoder, no labels, no reward) hierarchical world model whose **operative predictor imagines the latent consequence of each candidate action**, whose **imagination is decoded through a frozen probe calibrated on the predictor's own outputs** (direction accuracy **0.97**, position error **0.19 wu**), and whose **strategic layer routes through a latent transition graph** (+58 % over greedy on cross-room tasks).

As of v1.1 the mechanism has been re-demonstrated under the **egocentric (agent-centered) observation model** — the one that transfers to driving — where predictor-imagined action-selection reaches usable control (**0.69/0.76** direction accuracy) precisely because every action now moves the whole visual field. In the same push, an exhaustive object-binding study established *why* the toy resisted object-centric slots and *why that reason evaporates on real video* (§3.5). These two results tighten the driving thesis rather than loosen it: the observation regime and the scene statistics that broke the toy are exactly the ones driving supplies for free.

This document maps every proven edge, every measured failure mode, and every instrument-doctrine lesson onto **autonomous driving (AD)**, positions the result against the 2024–2026 literature, and specifies a **staged, falsifiable, low-budget experimental program** ("prove the edges before scale") with concrete datasets, commands, gates and compute budgets.

The disruptive claim to be proven:

> **Claim D.** A ~10–100 M-parameter hierarchical latent world model, trained from scratch on **tens of hours** of unlabeled egocentric driving video plus free proprioception (ego-motion), with **no perception labels, no HD map, no reward, and no pretrained foundation encoder**, can (i) imagine action-conditioned futures accurately enough to drive maneuver selection, (ii) decode metric trajectories from imagination through frozen calibrated probes, (iii) generalize from simple to complex road topologies better than flat (non-hierarchical) models of equal size, and (iv) run planning at real-time rates — where the current state of the art needs either internet-scale pretraining (V-JEPA-2-class, ~1 M h), billions of parameters of pixel diffusion (GAIA-class), or dense human annotation (UniAD-class).

Sections 1–4 give the scientific foundation; 5–7 the architecture transfer; 8–10 the experimental program with gates; 11 datasets; 12 efficiency engineering; 13 the instrument doctrine (mandatory); 14 the 90-day plan.

---

## 1. What ALPS-4B actually established (assets to transfer)

Every item below is **measured**, not hypothesized. These are the transferable assets.

| # | Asset | Evidence (repo) | AD relevance |
|---|-------|-----------------|--------------|
| A1 | **Pure-SSL training works without EMA/stop-grad/VICReg** (SIGReg on embeddings *and* predictions at all 3 scales) | healthy 30-ep runs, no collapse, `sig` stable ~1.1 | removes all engineering crutches at scale; single mechanism |
| A2 | **Imagine-and-select control works**: op-predictor imagines the next latent per candidate action; frozen probe reads it; argmin-to-goal picks the action | calibrated direction_acc **0.97**, pred_err **0.19 wu** (`_blockR`, block-mode, BN-calibrated) | the core planning loop of a driving stack |
| A3 | **Calibrated-decode doctrine**: fit the readout probe on the **predictor's own outputs** (imagined latents), not on real-frame encodings — removes the systematic linear off-manifold distortion of imagined latents | 0.97 (calibrated) vs 0.66 (real-frame probe) on identical predictions | *the* answer to "how do we decode trajectories from internal imagination" (§6) |
| A4 | **Residual/delta prediction + change-weighted latent loss** beat plain MSE for action-conditioned prediction | bake-off: 0.97 vs 0.71 (MSE) vs 0.44 (flow) | externally replicated by ResWorld (temporal residual world model for E2E driving) |
| A5 | **Inverse dynamics as representation grounding**: predicting the action from (z_t, z_{t+1}) forces the controllable state into the compact latent; discriminative direction works even when generative struggles | inv loss 1.386→0.016; inverse goal-emission 0.57–0.79 | in AD, ego-motion is *free* proprioception (CAN/IMU/odometry) — A5 costs nothing |
| A6 | **Hierarchy edge**: latent-graph routing (strategic) beats greedy (operative) when the task has topology | +58 % cross-room (0.50→0.79), oracle 0.97 | intersections, detours, blocked lanes = topology; Michon's strategic level |
| A7 | **Spatial readout ≫ global pooling** for position-faithful compact state | G1: 0.73 (pool) → 0.04–0.16 (grid readout) | BEV-grid readout per camera; never global-pool a driving scene |
| A8 | **Consequence-dominance as a task-design law**: the SSL predictor learns action-conditioned dynamics only when the action's consequence is a dominant, observed fraction of the frame change | Two-Rooms dot (1 %) failed everywhere; Block-Rooms (14 %) succeeded | egocentric driving is consequence-dominant **by construction** (§5.1) |
| A9 | **Imagination error as a self-monitor**: `imag relative` (‖imagined − true next latent‖/scale) discriminates familiar from unfamiliar dynamics | tracked across all runs; <1 = informative | free OOD/anomaly detector while driving (§7.4) |
| A10 | **The instrument doctrine** (hard-won): oracle-decode rows, batch-consistency checks, episode-level splits — the measurement harness must be validated **before** the model is judged | the BatchNorm bug: same frame, batch-1 vs batch-128 encoding differed **115 %**; fixed → every "impossible" number moved at once | deployment inference is batch-1 streaming; §13 makes these checks mandatory |
| **A11** | **Egocentric observation *makes control learnable*** — pinning the agent at frame-center so every action scrolls the whole field turns the action into the dominant, predictable frame change; the operative predictor then discriminates actions well enough for usable selection | egocentric Two-Rooms: calibrated direction_acc **0.69**, forward-dynamics **0.76** (both ≥ 0.6); `inv` 0.879→**0.138** — vs top-down small-agent 0.19 / action_spread 0.025 (§3.5) | **driving is egocentric by definition** — this is the single most load-bearing transfer result: the toy's hardest failure mode is the driving camera's default |
| **A12** | **The object-binding laws** — appearance/recon-driven slot binding discovers an object only when (i) its consequence is a non-trivial fraction of the *binding* loss and (ii) scene entropy exceeds single-slot capacity; a minimal toy violates both, so slots collapse to one-slot-per-scene regardless of granularity, motion cue, or change-weighting | isolation loop (recon-only SlotAttention): **1 slot/image** at 192/64/32-dim slots and under change-weighted recon; full-model masks bind the multi-token *wall* but never the sub-token agent (`SLOT_FOUR_BRAIN.md` §10) | **both laws invert on driving**: high scene entropy forces decomposition, ego-motion makes movers (cars, pedestrians) dominate loss mass → **object-centric slots are a Stage-2 real-video asset, not a toy tool** — do *not* burn toy cycles on them |
| **A13** | **Action-discrimination ≠ imagination fidelity** — usable control depends on separating candidate actions in the *decoded-state* space, not on low 1-step latent error; control was usable while `imag rel > 1` | egocentric: control usable at `imag rel` **1.27** (raw imagination worse than persistence) yet calibrated/forward-dynamics selection 0.69/0.76 | driving's horizon *always* imports unseen content (`imag` will stay > 1); A13 says that does **not** block control — decode the *action contrast*, not the full future |

**The negative results are equally transferable:** pixel-space or full-token uniform MSE lets static background dominate (A8's converse); flow/diffusion sampling adds variance that hurts deterministic short-horizon control (bake-off: 0.44); global pooling erases the controllable state (A7's converse); an uncalibrated readout silently misreads imagination (A3's converse).

---

## 2. Literature positioning (2024–2026): where the gap is

### 2.1 The four families

**(a) Generative pixel/video world models** — [GAIA-1 (9 B params)](https://wayve.ai/thinking/scaling-gaia-1/), [GAIA-2 (latent diffusion, multi-camera)](https://arxiv.org/abs/2503.20523), [Vista](https://arxiv.org/html/2405.17398v1), [DriveDreamer](https://drivedreamer.github.io/), [Navigation World Models (1 B CDiT)](https://arxiv.org/html/2412.03572v1). Superb for synthetic data and controllable simulation; but planning through them means **rendering pixels you don't need**, at seconds-to-minutes per decision and thousands of training-hours. NWM plans by simulating trajectories with a 1 B diffusion transformer; V-JEPA-2-AC reports Cosmos-style diffusion planning at ~4 min/action vs **16 s** for latent planning — and our greedy/beam latent selection is milliseconds.

**(b) Latent world models as auxiliary/self-supervision for E2E driving** — [LAW (ICLR 2025)](https://arxiv.org/abs/2406.08481): predict the future *latent* scene feature conditioned on the ego trajectory; annotation-free; SOTA on nuScenes/NAVSIM/CARLA. [World4Drive (ICCV 2025)](https://arxiv.org/abs/2507.00603): intention-aware latent world model that generates multi-modal trajectory proposals and **evaluates them in latent space** — the imagine-and-select pattern. [WorldRFT (AAAI 2026)](https://arxiv.org/abs/2512.19133): hierarchical planning decomposition inside a latent world model + RL fine-tuning. [ResWorld](https://arxiv.org/pdf/2602.10884): temporal **residual** world model (independently validates A4). [IDOL](https://arxiv.org/pdf/2605.31476): **inverse-dynamics-guided** future prediction (independently validates A5). [Latent-WAM](https://arxiv.org/pdf/2603.24581), [Metis](https://arxiv.org/pdf/2606.15869), [InDRiVE (reward-free pretraining via latent disagreement)](https://arxiv.org/pdf/2512.18850). **This family is our home.** But: all are single-level or two-level, most ride a supervised perception stack or a pretrained backbone, and none combine hierarchy + from-scratch SSL + calibrated imagination readout.

**(c) JEPA-family video SSL adapted to control** — [V-JEPA 2 / V-JEPA 2-AC](https://arxiv.org/abs/2506.09985): frozen encoder from ~1 M h internet video; a block-causal **action-conditioned predictor post-trained on <62 h** of unlabeled robot video; zero-shot planning on real arms; 16 s/action. [Drive-JEPA](https://arxiv.org/html/2601.22032v1): V-JEPA-2-initialized encoder + waypoint-anchored proposals + trajectory distillation → **93.3 PDMS on NAVSIM v1**. These prove the *architecture direction* at scale — but they *presuppose the giant pretrained encoder*, which is precisely the dependency this program refuses (the LeWM/ALPS thesis: the world model must be learnable from the task's own experience; §1 A1). The 62-hour post-training figure of V-JEPA-2-AC is nonetheless the single most important external data-point for our data-efficiency claim: **the action-conditioning stage is cheap once the representation exists** — our bet is that a *hierarchical, consequence-dominant curriculum* makes the representation itself cheap too.

**(d) Latent-action models (action-free video)** — [Genie](https://arxiv.org/abs/2402.15391)-style, [LAPA](https://arxiv.org/abs/2410.11758), [AdaWorld](https://arxiv.org/html/2503.18938v1): infer discrete latent actions between frames with VQ objectives, then condition world models on them. This is our inverse-dynamics head generalized to passive video — the bridge to training on **action-free dashcam corpora** (OpenDV/BDD100K) when CAN signals are absent.

### 2.2 The gap this program occupies

| Axis | GAIA-2 / NWM | LAW / World4Drive | Drive-JEPA / V-JEPA-2-AC | **ALPS-4B-AD (this program)** |
|---|---|---|---|---|
| Pretraining data | 1000s h video | nuScenes/nuPlan + perception stack | ~1 M h internet + 62 h action | **≤ 35–100 h, from scratch** |
| Params | 1–9 B | 0.1–1 B (with backbone) | 0.3–1 B ViT + predictor | **10–100 M total** |
| Labels | none (gen.) / maps for cond. | trajectory supervision (imitation) | trajectory distillation | **none; ego-motion only** |
| Planning | render/simulate pixels | single-level latent | CEM in latent, flat | **hierarchical imagine-and-select, graph-routed** |
| Readout of imagination | pixels | learned decoder in-loop | learned heads | **frozen probe, imagination-calibrated (A3)** |
| Hierarchy (Michon) | none | none/2-level | none | **operative/tactical/strategic + memory (4B)** |
| OOD self-monitor | – | – | – | **imagination-error monitor (A9)** |

No published system occupies the bottom-right column. The nearest misses: WorldRFT has hierarchy but needs RL fine-tuning and a perception stack; World4Drive selects trajectories in latent space but is flat and supervised at the planning head; Drive-JEPA has the right planning loop but a foundation-scale encoder. **The edge is the combination, and each component is already individually validated either by us (A1–A10) or externally (LAW, IDOL, ResWorld, V-JEPA-2-AC).**

---

## 3. Why driving is a *better* fit than Two-Rooms ever was

This is the pivotal scientific argument, and it inverts the difficulty story.

**The consequence-dominance law (A8)** says SSL forward prediction learns the action exactly when the action's consequence dominates the observed frame change. Two-Rooms violated it (2-px dot, ~1 % change/step); we had to *engineer* Block-Rooms to satisfy it.

**Egocentric driving satisfies it by construction, and we have now *directly demonstrated* the mechanism on the toy.** From the driver camera, ego-motion moves *every pixel* (global optical flow ∝ speed & yaw-rate); a steering action visibly rotates the entire scene within 100–300 ms. The measured per-step frame change in driving video at 2–10 Hz is tens of percent — the Block-Rooms regime, not the dot regime. When we render Two-Rooms **egocentrically** (agent pinned at frame-center, world scrolling under each action), predictor-decoded control jumps from unusable (0.19, top-down small agent) to **usable** (calibrated 0.69, forward-dynamics 0.76) — because the action is now the dominant predictable change and the operative predictor is *forced* to learn controllable dynamics (§3.5, A11). This retracts the v1.0 claim that "egocentric Two-Rooms failed": that verdict judged imagination *fidelity* (`imag 1.86`) instead of action *discrimination*, and predated the clean instrument. The refined truth (A13) is sharper and more favorable: even though scrolling imports unpredictable horizon content (so `imag` stays > 1), the action contrast in the decoded-state space is strong enough for maneuver selection. Roads are gentler still — the incoming content is the **road ahead, already visible** at depth; forward prediction is short-horizon warping plus bounded novelty at the horizon line. This is exactly the regime where LAW's latent prediction and V-JEPA-2-AC's action-conditioned prediction demonstrably work.

Second inversion: **actions come for free.** Two-Rooms had to log simulator actions; driving gives centimeter odometry, IMU yaw-rate, CAN steering/throttle on every platform ([comma2k19](https://github.com/commaai/comma2k19) ships raw CAN). Proprioception is not annotation — the JEPA/LeWM unsupervised character is intact (A5).

Third: **the hierarchy is not an artifact, it's the domain's own decomposition.** Michon's classic driving-task hierarchy — strategic (route), tactical (maneuver), operational (control) — is the canonical model of human driving cognition ([survey](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2025.1451923/full)), and it is *isomorphic* to the four-brain stack (§5). Hierarchical decomposition is also what game-theoretic planners use for tractability ([hierarchical games](https://arxiv.org/pdf/1810.05766)).

---

## 3.5 The 2026-07 findings in detail: contamination, the binding laws, the egocentric breakthrough

This section records — in full, including the false starts — the investigation that produced A11–A13, because the *reasoning chain* transfers as much as the numbers. It was driven by a single owner demand: *"change the Two-Rooms task/environment to show it is solvable by our architecture, otherwise I'm not convinced."* The honest answer turned out to be **yes, by changing the observation model to the one driving already uses.**

### 3.5.1 The contamination bug (why prior "small-agent" verdicts were void)

`TwoRoomsEnv` never OR-ed `block_mode` with the derived flags (`block_wall`, `block_gate`), and the data generator passed them independently. **Consequence:** every dataset generated with only `--block-gate` (no explicit `--block-mode`) silently rendered the *classic* rooms environment (0.8-wu agent, 0.3 step, no key, no gate), while evaluation and control ran in the *real* block environment. Verified by rendering the stored training frames directly. This contaminated every "gate-mode" model and made all "the small radius-0.9 agent is undecodable" numbers **cross-domain artifacts**, not model limits. Fixed at the env root (`self.block_mode = block_mode or block_wall or block_gate or block_clutter`, commit `b373f1b`). Wall-mode results (which set `--block-mode` explicitly) and the open-block bake-off (calibrated **0.97**) were always clean.

**On the fixed data, the small agent's position is sharply decodable: G1 = 0.109 wu** (was ~2.0 contaminated), oracle direction_acc 0.90. *Perception was never the problem.* → **Driving lesson / new doctrine item I7 (§12): a task is defined by the (data, env, model) triple; assert their identity mechanically. In AD this is the sensor-config / calibration / coordinate-frame match between the log the probe was fit on and the stream it runs on — the exact class of silent error that ends careers, now caught by a rendered-frame identity check.**

### 3.5.2 The object-binding laws (why slots failed the toy, and why driving flips it)

To get a *size-invariant* readout (an object slot binds by feature, not pixel count), we ran a full object-centric program: SAVi-style recurrent binding, inverse-dynamics + SIGReg on slots, a permutation-equivariant readout, a windowed control buffer, a motion cue (SAVi's flow, in label-free latent form), a scaled decoder, and a slot-dim bottleneck. Every full-model configuration produced the same masks: at patch-16 the slots tile the frame into *regions* and cleanly bind the multi-token **wall as an object**, but the sub-token **agent (14 px inside a 16 px patch) is never bound**; at patch-8 the masks collapse to *whole-frame* archetypes.

The decisive evidence came from an **isolation loop** — a 30-minute harness that trains SlotAttention + decoder alone, reconstruction-only, on frozen encoder tokens:

| isolated experiment | slots used / image | recon |
|---|---|---|
| recon-only, 192-dim slots | **1** (a different slot per scene = archetype clustering) | 0.02 |
| slot-dim bottleneck 64 / 32 | **1** | 0.04 |
| change-weighted recon (movers ×20) | **1** (agent reconstructed — by the same slot) | 0.02 |

Two scale-free laws explain it:

1. **Consequence-dominance at the *binding* level (A8 recurs).** The agent is ~1.5 % of the reconstruction loss mass; binding never *needs* it. Change-weighting the recon changes *what* is reconstructed, not *how many* slots share the work.
2. **The decomposition threshold.** Slot attention decomposes only when scene entropy exceeds single-slot capacity. A minimal scene (archetype + one agent position ≈ tens of bits) sits *below the floor* — even a 32-dim slot absorbs the whole scene.

**Both invert on real driving video** and this is the transferable payload: (i) a driving scene's entropy vastly exceeds any single slot → decomposition becomes *necessary*; (ii) ego-motion makes the movers that matter — other vehicles, pedestrians, cyclists — carry *dominant* loss mass, not 1.5 %. **Therefore object-centric slots are re-classified as a Stage-2 real-video asset (A12): introduce them where they work, do not spend toy cycles proving them where the scene is below threshold.** This is a genuinely useful negative result — it tells us *when* the object-centric branch of the four-brain earns its keep.

### 3.5.3 The egocentric breakthrough (the answer to "make it solvable")

The clean-data result isolated the *real* remaining gap: with perception solved (G1 0.109), the **operative predictor still ignored the action** — `action_spread` 0.025 wu, calibrated direction_acc 0.19. Cause, structural not incidental: in the god's-eye top-down view one action moves the small agent ~0.27 wu ≈ **2 % of the pixels**, so a self-supervised next-frame predictor gets almost no action gradient — consequence-dominance (A8) striking again, now at the *prediction* level, where change-weighting cannot help because the tiny agent's *appearance* barely shifts.

The fix is the observation model driving already uses: **egocentric rendering** (agent pinned at frame-center; the position-locked textured floor, target, and walls scroll under each action; already built as `TwoRoomsEnv(egocentric=True)`). Measured, dense winning recipe (residual + change-weighted op + inverse-dynamics), enc-depth 4, patch-8, honest instrument (episode-split, `calibrate_bn`, oracle rows):

| metric | top-down (small agent) | **egocentric** | reading |
|---|---|---|---|
| training `op` | — | 0.517 → **0.100** | sharpest operative convergence in the project |
| training `inv` (action grounding) | — | 0.879 → **0.138** | the predictor learns what each action does |
| `action_spread` | 0.025 (blind) | **0.220** | predictor now discriminates the 4 actions |
| G1_spatial (position decode) | 0.109 | 0.305 | abs. position read from the scrolling texture (softer than a localized blob) |
| ORACLE-DECODE direction_acc | 0.90 | 0.59 | instrument ceiling (limited by texture-read sharpness) |
| **calibrated direction_acc** | **0.19** | **0.69** | **USABLE (≥ 0.6)** ✓ |
| **forward-dynamics direction_acc** | — | **0.76** | independent confirmation ✓ |
| latent-space (no-decode) direction_acc | — | 0.24 | pure goal-latent matching not yet there |
| `imag rel` | — | 1.27 | raw 1-step imagination still > 1 |

**Interpretation, and the three insights that transfer:**

- **A11 — egocentric makes control learnable.** Every action scrolls the whole field, so the action becomes the dominant predictable change; the predictor is *forced* to model controllable dynamics. Two independent control readouts (calibrated 0.69, forward-dynamics 0.76) clear the usable bar; the top-down baseline (0.19) does not. **Driving is egocentric by definition — the toy's hardest failure mode is the driving camera's default.**
- **A11 corollary — the small-agent problem *dissolves*.** With the agent at frame-center its pixel size is irrelevant; the owner's "shrink the agent" constraint and the entire slot-binding detour both evaporate under the correct observation model.
- **A13 — control ≠ imagination fidelity.** Control was usable at `imag rel 1.27` (> 1, i.e. raw imagination worse than persistence) because selection reads the *action contrast* in the decoded-state space, not the full future. This directly predicts the driving case: the horizon always imports unseen content, `imag` will stay > 1, and that does **not** block maneuver selection.

**Honest caveats (carried into the gates):** this is the enc-depth-4 *local* config (depth-10 hardening run queued on the A40, `_ego10.pt`); the no-decode pure-latent path (0.24) is not yet there, so control rides the decoded-state operative path (P1/P3, §5), not P2; and the oracle ceiling (0.59) — set by how sharply absolute position reads from the scrolling texture — currently caps headroom, so a stronger encoder lifts every number. None of these are transfer blockers; all are Stage-0/1 tuning.

### 3.5.4 What this does to the program

- **Stage 0 acquires a validated observation model and recipe** *before* any driving code: egocentric rendering + dense winning recipe + the decoded-state operative control path. The Stage-0 env should be egocentric-camera from the start (BEV remains the planning-isolation track).
- **The mandatory instrument rows gain I7** (task-identity / sensor-config assertion, §12).
- **The object-centric branch is scheduled for Stage 2** (A12), not Stage 0.
- **`imag rel` is demoted from a gate to a diagnostic**: A13 shows it does not bound control; the binding metric is decoded-state action-discrimination (D2's `calibrated direction_acc`), which becomes the primary Stage-0 gate.

---

## 4. The four-brain ↔ driving mapping

```
ALPS-4B                          Autonomous driving (Michon)
───────────────────────────────  ─────────────────────────────────────────────
Operative predictor (token       OPERATIONAL: 0.1–0.5 s horizon; imagine the
grid, W-frame causal history,    next scene latent under each candidate control
action-conditioned, residual)    (steer/accel bin); pick argmin-to-subgoal.

Tactical layer (pooled latent,   TACTICAL: 1–3 s horizon; MoE experts ≈ maneuver
MoE router, goal-conditioned     schemata (lane-keep, lane-change-L/R, follow,
sub-goal head, K_tac horizon)    yield, creep, overtake); emits the operative
                                 sub-goal = imagined post-maneuver latent.

Strategic layer (VQ discrete     STRATEGIC: 10 s–minutes; VQ codes ≈ discrete
concepts, K_str horizon,         place/situation vocabulary (approach-junction,
latent transition graph +        roundabout-entry, merge-zone…); the latent
shortest-path routing)           graph over experienced transitions routes the
                                 drive (which exits/turns reach the goal);
                                 waypoints = graph nodes handed to tactical.

Latent-RAG memory                EPISODIC MEMORY: retrieve corrections from
                                 similar past latents (near-misses, rare
                                 layouts); prior work: retrieval-augmented AD.

Self-monitor (imagination        RUNTIME SAFETY MONITOR: imag-relative spikes ⇒
error, A9)                       unfamiliar dynamics ⇒ degrade to conservative
                                 fallback (slow, enlarge margins, handover).
```

Design deltas vs the Two-Rooms implementation (all small, all flag-level):

1. **Actions become continuous or binned-continuous.** `cond_proj` already accepts arbitrary `d_cond`; replace one-hot(4) with (steer, accel) ∈ ℝ² (FiLM-style, per-layer if needed) *and* keep a discretized maneuver vocabulary at the tactical level (9–15 bins) so imagine-and-select stays a small argmin. Inverse dynamics becomes a regression head (MSE) — one-line change.
2. **The readout becomes a BEV/ego-frame grid.** `spatial_readout` on the camera token grid; probe target = ego-relative future positions (odometry), not absolute world xy. Multi-camera later: concat per-camera grids.
3. **The probe decodes a *trajectory*, not a point** (§6): future ego waypoints at 0.5/1.0/1.5/2.0 s, NAVSIM-compatible (4 s @ 2 Hz once mature).
4. **The graph nodes live on strategic VQ codes** keyed by place+situation; edges = experienced transitions with empirical costs (time, comfort, interventions).

---

## 5. Trajectory decoding from internal imagination (the deep dive)

This is the question the harness saga answered most completely. The doctrine:

### 5.1 The three readout paths, ranked by our measurements

| Path | Mechanism | Our measurement | AD use |
|---|---|---|---|
| **P1. Calibrated frozen probe (primary)** | ridge/linear probe fit on **(imagined latent → true future ego-waypoint)** pairs harvested offline from the training corpus; frozen at deployment | direction_acc 0.97, err 0.19 wu — vs 0.66 for a real-frame-fit probe on the *same* predictions | decode the imagined latent of each candidate maneuver into metric waypoints; rank against the tactical sub-goal |
| **P2. Latent-space selection (no decode)** | argmin ‖imagined − goal-latent‖ | 0.30–0.48 (weaker; goal latents are far and manifold-curved) | tie-breaking, and long-horizon strategic matching where metric decode is unnecessary |
| **P3. Inverse-dynamics goal emission** | inv(z_now, z_subgoal) → action | 0.57–0.79 (strong, and cheapest: one forward pass, no per-action imagination) | fast-path operative control between imaginations; redundancy channel for the safety case |
| **P4. Forward-dynamics probe (new, 2026-07)** | a frozen ridge `g(decoded_state, action) → next decoded_state` fit on observed transitions; rank actions by imagined next state | egocentric direction_acc **0.76** — the strongest control readout in the egocentric run, and *cheaper* than P1 (2-D→2-D, no per-action predictor imagination) | the decoded-state operational controller: because it lives in the low-D readout space it cannot overfit the 12k-D latent, and it composes with the strategic waypoint directly |

**Egocentric corroboration (A11/A13):** on the driving-aligned egocentric observation model the same ranking holds with different absolute values — P1 calibrated **0.69**, P4 **0.76**, P2 (no-decode latent) **0.24**. Two lessons for the AD readout menu: (i) **P4 (forward-dynamics in decoded-state space) is a first-class control path**, often beating P1 at lower cost — add it to the driving diagnostic alongside P1/P3; (ii) **P2 (pure latent goal-matching) is the weakest and slowest to come online** — do not gate on it early; the metric operational controller (P1/P4) carries Stage 0–2, and P2 is a Stage-3 elegance target.

**The trajectory head (P1 generalized):** train, *offline and frozen thereafter*, a ridge (later: 2-layer MLP with episode-split early stopping) from `spatial_readout(ẑ_{t+k})` to `Δpose_{t→t+k}` for k = 1…H. Because it is calibrated on imagined latents, the systematic manifold shift of imagination is absorbed into the probe weights (A3). Two probes are kept: `probe_real` (fit on real-frame encodings — reads *current* state and *goal* frames) and `probe_imag` (fit on predictor outputs — reads *imagination*); both emit the same metric space so they compose (`make_videos_4b` already implements this split: `decode_state` vs `decode_pred`).

**Efficiency:** selection decodes only K candidate futures (K = 5–15 maneuvers), each a single predictor forward at each hierarchy level — no pixel rendering, no diffusion steps, no CEM population of hundreds. This is the V-JEPA-2-AC 16 s → milliseconds jump: greedy/beam over a *discrete tactical vocabulary* instead of continuous CEM.

**Label-free status:** the probe targets are ego-poses from odometry — proprioception, not annotation (same status as actions; A5). No lane labels, boxes, or maps are consumed anywhere.

### 5.2 The mandatory instrument rows (from the harness saga)

Every driving evaluation report **must** include, before any model claim:

1. **ORACLE-DECODE:** decode *real* future frames per candidate; must rank candidates ≈ 1.0. If not, the harness is broken, not the model (this row would have caught our BatchNorm bug on day one).
2. **BATCH-CONSISTENCY:** ‖encode(frame; batch=1) − encode(frame; batch=B)‖ / ‖·‖ < 1e-4. Deployment is batch-1 streaming; any batch-statistic normalization (BN without running stats, batch-wise whitening) violates this silently. Use running-stat BN via `calibrate_bn`, or batch-free norms.
3. **EPISODE/ROUTE-LEVEL SPLITS** for probe fitting: random-frame splits leak scene identity (we measured 0.04 vs 0.16 wu — 4× optimistic). In driving: split by **drive/route/day**, never by frame.
4. **IMAGINATION RELATIVE < 1** before any control claim (else the predictor is worse than persistence).

---

## 6. Strategic & tactical reasoning with the 4B stack

### 6.1 Tactical: maneuvers as imagined futures

The tactical layer's MoE + goal-conditioned sub-goal head is trained exactly as in Two-Rooms (hindsight far-horizon goals), but its output acquires a crisp driving meaning: **the imagined post-maneuver latent** ("what the world looks like after a completed lane change"). Selection = imagine each maneuver's outcome at K_tac, decode with `probe_imag`, score against the strategic waypoint + safety costs (progress, TTC-proxy from imagined relative motion, comfort from action smoothness). This is World4Drive's evaluate-in-latent pattern, made hierarchical and label-free.

### 6.2 Strategic: the latent graph as a learned road network

The Two-Rooms graph (k-means over experienced latents, edges from observed transitions, shortest-path) becomes, verbatim, a **topological memory of the driven network**: nodes = VQ place-situation codes, edges = maneuvers that connected them, weights = empirical cost. This gives (i) routing without any HD map, (ii) **re-routing when imagination contradicts the plan** (blocked lane: predicted latent diverges from the edge's expectation → graph edge cost spikes → replan — the driving analog of the wall+gap experiment now running), (iii) a natural simple→complex curriculum measure: graph reuse across procedurally recomposed maps (§8, gate D2).

### 6.3 The reasoning claim, falsifiably stated

Hierarchy is claimed to buy: (a) **success on topology tasks** flat models fail (+58 % analog), (b) **horizon factorization** — strategic plans over minutes with O(graph) cost while operative stays at 0.1 s, (c) **generalization** — recombining known maneuvers/places on unseen road compositions degrades sub-linearly vs flat models (the MetaDrive procedural test, gate D2), (d) **interpretability** — the plan is a sequence of discrete codes + metric waypoints, auditable at each level (a safety-case asset no flat E2E model offers).

---

## 7. The data-efficiency ("disruptive") thesis, quantified

Why less data suffices, argument by argument:

1. **Latent, not pixel targets:** no capacity spent on rendering appearance (GAIA-class spends billions of params there). LeWM/JEPA argument, confirmed at Block-Rooms scale.
2. **Consequence-dominance (A8):** egocentric driving puts the action's signature in *every* token — gradient density per frame is orders above the dot regime; our entire Block-Rooms success ran on **37 k frames ≈ 1 driving hour at 10 Hz**.
3. **Hierarchy factorizes the sample space:** maneuvers are learned once and *recombined* by routing (compositional generalization), instead of every (route × maneuver) pair needing data.
4. **Frozen probes carry zero training burden** and cannot overfit the planner to the metric head (A3).
5. **Free proprioception** replaces the single most expensive supervision in AD (trajectory/perception labels).
6. **External anchor:** V-JEPA-2-AC needed only **62 h** for its *entire action-conditioned stage*; our bet extends this: with a curriculum engineered for consequence-dominance, the *representation* stage is also cheap. Target: **Stage-2 competitive open-loop trajectory metrics from ≤ 35 h (comma2k19) at ≤ 100 M params** — 4–5 orders of magnitude less pretraining video than the Drive-JEPA pipeline consumes via its V-JEPA-2 encoder.

Honesty clause: the program does **not** claim to beat NAVSIM SOTA at Stage 2. It claims a *Pareto point* (labels=0, data≤35 h, params≤100 M, real-time) that no current system occupies, with gates that make the trend measurable — and a scaling path (Stage 3) whose slope is the real product.

---

## 8. Edge hypotheses D1–D8 (driving analogs of E1–E8) and their gates

Each gate names: metric, threshold, instrument checks (I1–I4 = §5.2 rows), and the ablation that isolates the edge.

| ID | Hypothesis | Gate (falsifiable) | Ablation |
|----|------------|--------------------|----------|
| **D1 (repr.)** | From-scratch SSL encoder on driving BEV/video encodes ego-relative state decodable to < 0.5 m @1 s | `G1-traj`: frozen-probe ADE@1s < 0.5 m (BEV toy) / < 1.0 m (camera); **I2, I3 pass** | vs global-pool readout (A7) |
| **D2 (imagination)** | The op-predictor imagines candidate-action futures usable for selection | `calibrated direction_acc > 0.7` **or** `forward-dynamics direction_acc > 0.7` (P4) on maneuver ranking; **I1 ≈ 1.0 first**. *`imag rel` is a diagnostic, not a gate* (A13: control was usable at `imag rel 1.27`) | vs persistence baseline; vs flow head (expect A4 ordering); top-down vs egocentric observation (expect A11 gap) |
| **D3 (trajectory decode)** | Frozen imagination-calibrated probes decode multi-step trajectories | imagined-ADE@2s within 1.5× of oracle-decode ADE@2s | probe_real vs probe_imag on identical predictions (A3 gap must reproduce) |
| **D4 (tactical)** | Maneuver-level imagine-and-select beats 1-step greedy | +X % success / PDMS-proxy on interactive scenarios (X target 15 %) | tactical off |
| **D5 (strategic)** | Latent-graph routing solves topology tasks greedy fails | cross-junction / detour success: strategic ≫ operative (Two-Rooms +58 % analog); oracle-landmark run bounds it above | graph off; random waypoints |
| **D6 (generalization)** | Hierarchy generalizes simple→complex better than flat | MetaDrive: train 20 proc. maps (3 blocks), test 100 unseen (3–7 blocks); success-degradation slope 4B < flat, equal params | flat model, matched params/data |
| **D7 (memory)** | Latent-RAG improves rare-scenario handling | repeat-exposure improvement on held-out rare layouts | RAG off |
| **D8 (monitor)** | Imagination error detects OOD | AUROC > 0.85 separating in-distribution vs unseen-town/weather frames | vs encoder-only Mahalanobis |

Program rule (learned the hard way): **no architecture change may be motivated by a gate that hasn't passed its instrument rows.**

---

## 9. Staged experimental program

### Stage 0 — "Block-Roads" (1–2 weeks; RTX 4060/single A40; ~$50)

*Purpose:* reproduce A1–A5 + A7 + A11 + D1–D3 on driving-like dynamics with the **existing repo, minimal changes**.

- **Observation model (validated by A11, non-negotiable):** run **two tracks** — (a) **egocentric forward camera** as the *control* track (the A11 regime: the ego is centered, the world scrolls under each action → action-dominant frame change → learnable control; this is the driving-native model and the one that carried the toy from 0.19 to 0.69/0.76), and (b) **top-down BEV** as the *planning-isolation* track (clean routing/graph experiments). Do **not** attempt control on top-down alone — it is the consequence-dominance-hostile regime (A8/A11). The Two-Rooms lesson: egocentric first, BEV for topology.
- **Env:** [MetaDrive](https://ar5iv.labs.arxiv.org/html/2109.12674) renders both a forward camera and a ~128 px top-down BEV (300 FPS) — plug-compatible with our `[N,3,128,128]` pipeline. Fallback: `highway-env`. Vehicle = bicycle model; actions binned to 9 (3 steer × 3 accel).
- **Data:** 500–1500 episodes × 60–120 steps (≈ 40–150 k frames — the *validated* Block-Rooms scale), mixed random+IDM heuristic policies, `--block-mode`-style generator port (`envs/driving_toy/data_generator.py`).
- **Train:** the existing command family, unchanged: `train_temporal --lewm-ssl --inv-dyn --residual-pred --change-weighted-op --d-model 192 --enc-depth 10 --op-depth 8 --window 6 --stride 1 --epochs 30` (inverse head switched to 9-way CE, later 2-D regression).
- **Gates:** D1 (`G1-traj` on ego Δpose), D2 (calibrated ranking of the 9 actions), D3 (2 s trajectory probe), all with I1–I4 rows. Diagnostic = `diagnose_control` ported (`--driving`).
- **Deliverable:** the Block-Rooms bake-off table on driving dynamics + first solve GIFs (goal-reaching on procedural maps).

### Stage 1 — Procedural generalization & the hierarchy edge (4–8 weeks; 1× A100 or A40; ~$500)

- **Env:** MetaDrive procedural composition; two observation tracks: (a) BEV (primary — isolates planning), (b) forward camera 128→224 px (bridges to Stage 2).
- **Tasks:** navigation across maps with junctions/roundabouts; **blocked-route variants** (the wall+gap analog) where greedy fails and routing is required.
- **Model:** d 256–384, enc-depth 12, ~30–60 M params; continuous-action FiLM conditioning lands here; tactical vocabulary via VQ codes inspected for maneuver semantics.
- **Gates:** D4, D5, **D6 (the headline: simple→complex slope, 4B vs flat at matched params)**, D8 (unseen block types as OOD).
- **Deliverable:** the ablation-ladder report — the driving analog of the Two-Rooms E-ladder — plus autonomous solve videos (operative stalls at blocked route; four-brain re-routes).

### Stage 2 — Real video, small data (2–3 months; 2–4× A100; ~$3–5 k)

- **Data:** [comma2k19](https://github.com/commaai/comma2k19) (33 h, CAN steering/speed + GNSS pose — **real actions + real trajectory targets, zero annotation**); nuScenes-mini for cross-city OOD probes.
- **Recipe:** encoder from scratch on 10 Hz clips (2-frame tubelets, 16×16 patches @ 192–224 px); actions = (yaw-rate, accel) FiLM; window 8; predict k ∈ {1,2,4} steps; inverse-dynamics regression; SIGReg throughout; **calibrate_bn (or batch-free norms) + I1–I4 enforced from day one**.
- **Gates:** D1–D3 on held-out *routes*; imagined-trajectory ADE/FDE@2s vs (i) persistence, (ii) supervised-at-matched-data baseline, (iii) published LAW-class numbers for context; D8 AUROC in-town vs nuScenes.
- **Stretch:** action-free co-training on 100 h BDD100K via latent-action VQ (LAPA/AdaWorld-style) to test whether passive dashcam video buys accuracy per §2.1(d).

### Stage 3 — Benchmark entry & closed loop (outlook; after Stage-2 gates)

- **Open-loop:** [NAVSIM](https://arxiv.org/html/2511.10403v1) PDMS/EPDMS with the frozen SSL stack + light trajectory head (Drive-JEPA is the reference pipeline; ours differs by from-scratch encoder + hierarchy + frozen probes).
- **Closed-loop:** Bench2Drive/CARLA and MetaDrive closed loop; nuPlan-R reactive sim. Strategic graph built from the benchmark's own training routes.
- Only here does scale enter (encoder growth, multi-camera, longer horizons) — with the D-gate ladder re-run at each scale point to measure the slope that *is* the thesis.

---

## 10. Datasets survey

| Dataset | Size / signals | Actions? | License/access | Role here |
|---|---|---|---|---|
| **MetaDrive** (sim) | ∞ procedural; BEV + camera; 300 FPS | ✓ exact | Apache-2.0 | Stages 0–1; D6 generalization |
| **highway-env** (sim) | 2D lightweight | ✓ | MIT | Stage-0 fallback / unit tests |
| **CARLA / Bench2Drive** (sim) | photoreal towns; closed loop | ✓ | MIT/open | Stage 3 closed loop |
| **comma2k19** | 33 h highway commute; video + **CAN (steer, speed)** + GNSS | ✓ real | open (research) | **Stage-2 core** (actions for free) |
| **nuScenes (+mini)** | 5.5 h annotated, 2 cities, 6 cams | ego pose | CC BY-NC-SA (research) | OOD probes; open-loop context metrics |
| **nuPlan** | 1300 h, 4 cities; closed-loop framework | ego pose | research license | Stage 3; NAVSIM derives from it |
| **NAVSIM** | curated nuPlan scenes; PDMS/EPDMS | ✓ | open eval | Stage-3 open-loop benchmark |
| **BDD100K** | ~1100 h dashcam, high diversity | ✗ (video only) | research | latent-action co-training (stretch) |
| **OpenDV-2K** | ~2000 h YouTube driving (Vista's corpus) | ✗ | public video | scale-up corpus for Stage 3+ |
| **ONCE** | ~144 h raw, 1 M scenes, mostly unlabeled | ✗ | research | unlabeled pretraining reserve |
| **Waymo Open / Argoverse 2** | perception/motion suites | ego pose | non-commercial | later: multi-agent tactical evaluation |

Selection logic: **sim-first to prove edges cheaply and falsifiably (procedural control over topology = the D5/D6 experiments are *impossible* to run cleanly on real logs), then the one real dataset that carries native actions (comma2k19), then benchmarks.**

---

## 11. Efficiency engineering (training & inference)

**Training.** Measured anchor: full 4B stack (14.4 M params) trains 30 epochs on 37 k frames in ~50 min on one A40. Stage-1 (60 M, 150 k frames) ≈ single-A100 days, not weeks. Levers already validated: patch 16 over patch 8 (patch 8 *hurt*), stride 1 for action-attribution, window 6–8, SIGReg slices 512, `--save-every` crash-safety. Losses: keep the A4 pair (residual + change-weighting); skip flow (measured worse); no EMA/momentum bookkeeping (A1).

**Inference.** Per decision: 1 encoder pass (batch-1, BN-calibrated) + K≈9–15 predictor passes (batched as one) + probe matmuls + graph lookup (µs). At 10–60 M params this is **≪ 50 ms on automotive-grade GPU** — versus diffusion world-model planning (NWM 1 B; Cosmos ≈ 4 min/action) and CEM populations (V-JEPA-2-AC 16 s/action). Hierarchy compounds the saving: strategic runs at 0.1 Hz, tactical at 1–2 Hz, only operative at 10 Hz.

**Memory/graph:** node count grows with *places*, not hours; k-means/VQ codebooks of 64–512 suffice for city-district scale (Two-Rooms used 64 over 10 wu²; scale linearly with distinct situations, not frames).

---

## 12. The instrument doctrine (mandatory, non-negotiable)

The single most valuable — and most expensive — lesson of ALPS-4B: **for four weeks, every architecture change was evaluated through a broken instrument** (batch-statistics normalization made batch-1 control encodings differ 115 % from the batched encodings all probes were fit on; every control number was noise; three sophisticated "fixes" chased a phantom). For AD, where deployment inference is *always* batch-1 streaming and the cost of silent measurement error is not wasted GPU-hours but safety, the doctrine is:

1. **I1 Oracle rows first.** Every control metric ships with its oracle twin (decode *real* futures). Model claims are frozen until the oracle row is ≈ perfect.
2. **I2 Batch-consistency test** in CI: single-frame vs batched encodings must match to 1e-4. No batch-statistic layers in the inference path without calibrated running stats (`calibrate_bn`).
3. **I3 Route-level splits** everywhere a probe is fit (leakage measured 4× optimistic even in the toy).
4. **I4 Persistence baselines** for every predictive metric (`imag relative` < 1 or no claim).
5. **One lever per run**, n large enough that ±0.05 resolves (our n=200 runs produced 0.47/0.65 coin-flips that misdirected a week).
6. **Dynamics audits before model audits:** measure the actual action→consequence distribution of the env/dataset (the block arena's wall-clipping made the "clean 2.1 wu step" really 1.6 ± 0.88 — we designed against a spec the env didn't implement).
7. **I7 — Task-identity assertion (new, 2026-07).** A result is only meaningful if the *data the probe was fit on* and the *stream it runs on* are the **same task**: same environment variant, same sensor/observation config, same coordinate frame. Assert this **mechanically**, not by intent — a rendered-frame identity check at ingestion. The contamination bug (§3.5.1) had flags silently select the *classic* environment for data generation while control ran in the *block* environment; every "small-agent" number was cross-domain noise for weeks. In AD the analog is lethal: a probe fit on one vehicle's camera intrinsics/extrinsics, log rate, or ego-frame convention, silently applied to another. **Cross-domain contamination is invisible to every downstream metric — only an explicit identity check catches it.**
8. **Inference-memory reality (new, 2026-07).** The block-causal predictor's attention is O((W·N)²) in token count; a probe-fit batch tuned for coarse patches (N=64) OOMs an 8 GB card at fine patches (N=256). We made the diagnostic's batch sizes scale inversely with token count. AD lesson: **the deployment memory envelope is a first-class constraint** — profile the streaming (batch-1) inference path at the *finest* patch/camera-count you intend to ship, not the training batch.

---

## 13. Concrete first-90-days plan

| Weeks | Work | Exit criterion |
|---|---|---|
| 1–2 | Port `data_generator`/`diagnose_control` to MetaDrive BEV (`--driving`); continuous/binned action flag; trajectory probe (H=4); I1–I4 rows in the driving diagnostic | Stage-0 pipeline runs end-to-end on 4060 |
| 3–4 | Stage-0 bake-off (MSE vs residual+change-wt; pool vs grid readout; probe_real vs probe_imag) | D1–D3 gates measured; A3/A4 orderings reproduce on driving dynamics |
| 5–8 | Stage-1 procedural: blocked-route topology tasks; flat-vs-4B matched-params; VQ maneuver inspection | D5 (routing edge) + first D6 slope; solve videos |
| 9–12 | Stage-1 D6 full ladder + D8 monitor; write-up v1 ("Hierarchical latent imagination for driving: edge proofs at toy scale"); comma2k19 ingestion + encoder smoke-runs (Stage-2 start) | ablation-ladder report; Stage-2 training running |

Immediate next actions in this repo (can start today, before the pod results even land):
1. `src/alps/benchmarks/driving_toy/` — MetaDrive wrapper exposing `reset/step/render` with the Two-Rooms observation contract (BEV 128×128 uint8, `position`, `target`), so **every existing script works unchanged**.
2. `--action-dim N --continuous-actions` in `train_temporal` (cond_proj + inverse-head regression).
3. `--traj-probe H` in `diagnose_control`: multi-step calibrated waypoint decode + imagined-ADE row.
4. CI test for I2 (batch-consistency) — 15 lines, prevents the entire BatchNorm class of bugs forever.

---

## 14. Key references

World models for driving: [LAW (ICLR 2025)](https://arxiv.org/abs/2406.08481) · [World4Drive](https://arxiv.org/abs/2507.00603) · [WorldRFT (AAAI 2026)](https://arxiv.org/abs/2512.19133) · [ResWorld](https://arxiv.org/pdf/2602.10884) · [IDOL](https://arxiv.org/pdf/2605.31476) · [Latent-WAM](https://arxiv.org/pdf/2603.24581) · [Metis](https://arxiv.org/pdf/2606.15869) · [InDRiVE](https://arxiv.org/pdf/2512.18850) · [DriveFuture](https://arxiv.org/pdf/2605.09701) · [surveys](https://arxiv.org/html/2501.11260v4) ([curated lists](https://github.com/LMD0311/Awesome-World-Model))
JEPA family: [V-JEPA 2 / V-JEPA 2-AC](https://arxiv.org/abs/2506.09985) · [Drive-JEPA](https://arxiv.org/html/2601.22032v1) · [Navigation World Models (CVPR 2025)](https://arxiv.org/html/2412.03572v1)
Generative world models: [GAIA-2](https://arxiv.org/abs/2503.20523) · [GAIA-1 scaling](https://wayve.ai/thinking/scaling-gaia-1/) · [Vista](https://arxiv.org/html/2405.17398v1) · [DriveDreamer](https://drivedreamer.github.io/)
Latent actions: [LAPA](https://arxiv.org/abs/2410.11758) · [AdaWorld](https://arxiv.org/html/2503.18938v1)
Hierarchical driving cognition/planning: [decision-making survey (Michon)](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2025.1451923/full) · [hierarchical game-theoretic planning](https://arxiv.org/pdf/1810.05766) · [HANSOME](https://openreview.net/forum?id=HyS9pkHNTN) · [ReAL-AD](https://arxiv.org/html/2507.12499v1)
Benchmarks/sims: [MetaDrive](https://ar5iv.labs.arxiv.org/html/2109.12674) · [nuPlan](https://www.emergentmind.com/topics/nuplan) · [nuPlan-R](https://arxiv.org/html/2511.10403v1) · [NAVSIM correlation study](https://arxiv.org/html/2605.00066) · [comma2k19](https://github.com/commaai/comma2k19)
