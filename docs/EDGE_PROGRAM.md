# ALPS-4B Edge Program — proving the Four-Brain hierarchy at world-class standard

*Status: 2026-06. Supersedes the open items of `VALIDATION_PLAN.md`; keeps its gate*
*discipline. Every claim below is paired with a falsifiable gate and a compute target*
*(LOCAL = user GPU, light runs; A40 = RunPod, stable numbers).*

---

## 1. Where we stand (honest deep review)

### Proven (gate-validated)
| Claim | Evidence |
|---|---|
| Latent encodes state decodably (G1) | 0.19 wu (A40, supervised arm), threshold 0.30 |
| Actions move the latent correctly (G2) | sensitivity 18.6, directional consistency 1.00 |
| No representation collapse | eff-rank 9–17, dead-dims 0, pairwise-cos ≤ 0.31 |
| Strategic VQ concepts are room-level | concept→room probe 0.99 (A40) |
| System-2 latent-graph edge over System-1 | cross-room 0.07→0.36–0.40 (≈5×), monotone toward oracle |
| Frozen-latent + post-hoc probe protocol | encoder eval/no-grad everywhere; probes train only probe params |
| Latent-RAG generalizes without interference (offline) | +22% on unseen surprise contexts, 0% interference (surprise-gated) |
| Complex benchmark well-posed | BFS oracle 1.00 (hazards off); operative structurally 0.00 |
| Key state linearly identifiable | has_key probe 0.997–0.998 |
| Key-routed planning | semantic graph + key landmark routes start→key→goal (path-verified) |

### Open gaps (the program targets exactly these)
1. **G1_ssl unknown** — the pure self-supervised decode number (the LeWM
   linear-identifiability foundation) has never been measured at scale. Gate 5b is
   wired; needs the A40.
2. **The learned tactical module does no control work.** Kt=2 prediction decodes to
   ≈current position (too myopic to be a waypoint). The current "tactical" control
   tier is a *fine latent graph* — a fair abstraction-resolution tier, but not the
   learned MoE/predictor doing the work. The architecture claim needs the learned
   module in the loop.
3. **Complex closed loop = 0 locally** — decode precision (~0.7–0.8 wu) exceeds the
   key pickup radius (0.55 wu) at d128/local-data scale. Capacity-bound, not
   logic-bound; unverified until the A40 run.
4. **Supervision leak in the planner:** the explicit key landmark is built from
   dataset `has_key` labels. The encoder is label-free, but the *graph builder*
   isn't. For the unsupervised claim to be end-to-end, key discovery must be
   label-free (surprise events / VQ code flips).
5. **Strategic planning uses k-means landmarks, not the learned VQ codes.** The VQ
   bottleneck is validated as representation only; the discrete concept graph
   should be the planning substrate.
6. **Local numbers are noise-limited** (≈80-episode corpus → unstable graph;
   cross-room swings 0.08–0.40). All headline numbers must come from the A40
   (3000 episodes, 200-episode eval).
7. **No complex-mode proof videos** yet (simple-mode videos exist).

### Where the field is (June 2026) and where our edge sits
- **DINO-WM** (ICML 2025): frozen pre-trained visual features + ViT dynamics +
  zero-shot MPC. Validates *frozen-latent planning* — but flat (System 1 only),
  and rides a giant internet-pretrained encoder.
- **V-JEPA 2 / V-JEPA 2-AC** (Meta 2025): SSL video pretraining + action-conditioned
  post-training + MPC on real robot arms. Validates *SSL world models plan* — but
  flat, continuous-only, 1M-hour pretraining.
- **Hierarchical Planning with Latent World Models** (2026): plans abstract latent
  actions whose predicted states act as subgoals; reports compounding-error
  brittleness — exactly the failure our graph-anchored hierarchy avoids.
- **World Model as a Graph (L3P), SPTM/TSGM/FTM line**: latent landmarks + graph
  planning — but reward/Q-distilled or RL-coupled, not reward-free offline SSL.
- **Director** (2022): hierarchical control in world-model latent space — needs
  rewards and online RL.

**The unclaimed niche ALPS-4B targets:** a *small* (≈5M encoder), fully
self-supervised, reward-free, offline-trained world model whose **three tiers are
separately gate-validated**, with a **discrete (VQ) interpretable strategic plan**,
**graph-anchored** (no compounding rollout), plus **surprise-gated episodic
memory** — and the demonstration that the hierarchy edge **grows with task
horizon** while flat System-1 control decays to zero. No published system combines
these; that combination is the disruptive edge, and every element is testable.

---

## 2. Hypothesis ladder (each falsifiable; each one experiment)

| ID | Hypothesis | Gate (PASS) | Falsified if | Compute |
|---|---|---|---|---|
| **H0** | Pure-SSL latent linearly identifies state (LeWM) | G1_ssl < 0.30 wu; gap to SUP < 0.10 wu | SSL FAILs after epoch/SIGReg sweep | A40 (stage 5b) |
| **H1** | Strategic graph edge replicates at scale, SSL encoder | cross-room: graph ≥ 2× operative, n=200 | edge < +0.10 abs | A40 (stage 6) |
| **H2** | *Learned* tactical emitter earns its keep: coarse graph + tactical ≥ fine graph ≫ coarse alone | success(coarse+tac) ≥ 0.9·success(fine); fallback rate < 30% | emitter no better than coarse alone | LOCAL dev → A40 |
| **H3** | Strategic plan over *learned VQ codes* matches k-means landmarks | VQ-graph success ≥ 0.9·kmeans-graph; code→room purity ≥ 0.9 | VQ routing ≪ k-means | LOCAL dev → A40 |
| **H4** | Key acquisition is discoverable **label-free** (surprise spike / VQ flip at pickup) | detected landmark within 0.5 wu of true key; routing success = labeled variant | detector misses/false-positives dominate | LOCAL dev → A40 |
| **H5** | Complex (key→door→goal) closed loop solved by hierarchy at scale | operative ≈ 0 ≪ strategic < tactical; tactical ≥ 0.5; oracle 1.0 | tactical < 0.3 at d192/3k-eps | A40 (stage 7) |
| **H6** | **Edge grows with task horizon** (N-room mazes, N=2,4,6,8) | operative → 0 as N grows; hierarchy ≥ 0.5 at N=8; monotone edge curve | hierarchy decays like operative | LOCAL env-dev → A40 |
| **H7** | RAG-in-the-loop self-learning: post-WRITE success ↑ on perturbed layouts, 0 interference | +≥10% on perturbed; ≤2% drop nominal | interference returns | LOCAL dev → A40 |

**Kill criteria are real:** if H2 fails after horizon sweep, the paper reports the
tactical layer as *resolution tier* (fine graph), not learned module — no
overclaiming. If H0 fails, control results are reported on the SUP arm with the SSL
gap stated. Every fallback is still a publishable honest result.

---

## 3. Workstreams and design

### WS-A — Foundation at scale (H0, H1) — *A40, already wired*
Run `scripts/run_a40_validation.sh` (SSL default, G1_COMPARE=1). Read
`g1_identifiability.json` first. This is the go/no-go for the unsupervised story.

### WS-B — Learned tactical emitter (H2) — *the architecture-critical fix*
Why Kt=2 failed: 2 op-steps ≈ 0.8 wu — under waypoint resolution; the predictor
correctly predicts "almost here".
1. Retrain with tactical horizon sweep **Kt ∈ {4, 6, 8}** (+ multi-horizon heads),
   goal-conditioned on the target strategic concept (already implemented).
2. Inference: emitter proposes waypoint = decode(tac_predict(h_hist, c_next)).
   **Validity gates** before use: (i) *progress* — decoded proposal strictly closer
   to the target landmark than current position; (ii) *reachability* — proposal
   within REACH of a fine-graph node adjacent to the current cell. Invalid →
   fall back to graph waypoint; **log fallback rate** (it is part of the gate).
3. Ablation tiers become: operative / coarse graph / coarse+tactical-emitter /
   fine graph (upper anchor) / oracle.
LOCAL: d128, complex BFS-demo data (trains well locally), Kt sweep ≈ 6 min/run.
A40: best Kt at d192, n=200.

### WS-C — VQ concept-graph planning (H3)
Nodes = VQ code indices (64); edges = observed code transitions; node position =
frozen-probe decode of code embedding. Plan = Dijkstra over code transitions →
decoded waypoints. One new builder (`build_graph_vq`) + one ablation rung.
Deliverable doubles as interpretability: the plan is a readable sequence of
discrete concepts ("room-0 → door → room-1 → key → …").

### WS-D — Label-free key discovery (H4) — *closes the supervision leak*
Detector candidates (both label-free, frozen model):
1. **Surprise**: operative 1-step prediction error spikes at pickup (the frame
   appearance changes discontinuously — key vanishes).
2. **VQ flip at constant position**: strategic code changes while decoded position
   barely moves — a *state* (not location) transition.
Validation: precision/recall of detected events vs ground-truth pickups (labels
used **only for measurement**); then rebuild the key landmark from detections and
re-run H5 routing. Also powers the RAG write-trigger (same surprise machinery).

### WS-E — Complex closed loop at scale (H5) — *A40 stage 7, wired*
d192/depth10 decodes ≈0.19 wu ≪ 0.55 pickup radius → the local precision blocker
disappears by construction if H0/H1 hold. No controller hacks permitted; the only
allowed knobs are model scale, data density near the key (BFS demos), `--fine-k`.

### WS-F — N-room scaling curve (H6) — *the disruptive figure*
1. Extend `environment.py` to procedural N-room mazes (rooms in a chain/grid,
   doors between adjacent rooms, optional keyed doors). The env already supports
   4-room + keys; generalize layout generation (LOCAL, pure code).
2. Data: BFS-optimal + random mix per N (generator exists; parameterize).
3. Run the three-tier ablation per N ∈ {2,4,6,8}. Plot success vs N per tier.
**Expected headline figure:** operative decays → 0 by N=4; hierarchy stays ≥ 0.5
at N=8. This converts a point result into a *scaling law of the edge* — the claim
that makes the work matter beyond a toy: hierarchy is what survives task horizon.

### WS-G — RAG in the loop (H7)
After nominal training, WRITE corrections on a perturbed layout family (e.g.,
moved door), then closed-loop eval: perturbed (should ↑), nominal (must not ↓).
Surprise-gated retrieval only (validated offline recipe).

### WS-H — Proof videos (all hypotheses)
All rendered from A40 checkpoints; rendering itself is LOCAL-light (PIL→ffmpeg
path is proven; GIF→MP4 must go via PIL frames, not the GIF demuxer):
1. **Complex side-by-side** (the money shot): operative stalls at the locked door
   vs Four-Brain fetches key → unlocks → goal. Overlays: graph nodes/edges, the
   planned path, current target landmark, decoded-vs-true position trace,
   key-pickup freeze-frame.
2. **Tactical emitter view**: coarse landmark (far, behind wall) + emitted rough
   region (door) updating live (H2).
3. **VQ concept strip**: active code timeline under the frames — the discrete plan
   reading out as the agent moves (H3).
4. **Surprise trace**: prediction-error curve spiking exactly at key pickup (H4).
5. **Scaling montage**: N=2→8 mazes, operative vs hierarchy (H6).
6. **SSL foundation**: same episode decoded by SSL-probe vs SUP-probe overlay (H0).

---

## 4. Compute plan

| Resource | Used for |
|---|---|
| **LOCAL GPU** (light) | env extension (N-rooms), Kt-sweep prototypes (d128, complex BFS data ≈6 min/run), VQ-graph + detector prototyping, video rendering, smoke tests |
| **A40 RunPod** (one pod, resumable script) | stages 1–7 of `run_a40_validation.sh` (SSL default + G1 compare), then per-workstream confirm runs: H2 best-Kt @ d192, H4 relabeled complex, H6 per-N runs |

A40 sessions (each independent, artifacts pushed back as JSON/figures via git):
1. **Session 1 (~6–10 h):** full pipeline as wired → H0, H1, H5 first read.
2. **Session 2 (~4 h):** H2 (best Kt) + H3 + H4 confirm at d192.
3. **Session 3 (~6 h):** H6 scaling curve (4 maze sizes) + final video checkpoints.

## 5. Order of execution
1. **A40 Session 1 now** (nothing blocks it; H0 is the foundation read).
2. In parallel LOCAL: WS-B Kt sweep + WS-D detector + WS-F env (independent of A40).
3. A40 Session 2 after WS-B/C/D code lands.
4. A40 Session 3 (scaling + videos) last — it is the headline and needs the
   stabilized stack.
5. Paper update after each session: results land in §5 with gate tables; scope
   section shrinks as gates flip from pending → measured.

## 6. Writing standard
Every number in README/paper traces to a JSON artifact in
`results/two_rooms/validation/` produced by a committed script with a fixed seed.
Claims keep three statuses: **measured** (gate PASSED, artifact linked),
**pending** (gate wired, awaiting compute), **design hypothesis** (math only, no
empirical claim). The 42-advantage legacy list stays demoted until each line maps
to one of the three.
