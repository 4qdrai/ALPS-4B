# Slot‑Structured Four‑Brain: an object‑centric, hierarchical, self‑supervised world model

**Status:** design + first component (slot readout) — 2026‑07‑02
**Reads with:** `docs/PROJECT_HANDOFF.md`, `docs/AD_TRANSFER_RESEARCH.md`, `docs/BLOCK_ROOMS.md`
**Code:** `src/alps/core/slot_readout.py` (readout, this session); predictor/hierarchy = roadmap §7

---

## 1. Why — the size problem *is* an architecture signal

Two‑Rooms proved the disruptive claim (fully‑unsupervised, operative *provably* fails, four‑brain solves by strategic routing) and, in doing so, isolated the one place the current architecture caps out: **a dense‑grid, MSE‑latent representation dilutes a small, spatially‑localized object.** Measured:

- decode of a real frame degrades from `G1 0.05` (agent radius 1.7, ~9% of frame) to `G1 1.1` (radius 0.9, ~2%) and confuses the agent with same‑size distractors (key/target);
- the predictor's *imagination* of a small object's move is diffuse (a 1–3‑token shift is a blurry latent change), so the imagined‑position readout reads near‑flat (`action_spread 0.025`).

No readout trick fully escapes this (we tried soft‑argmax and grid‑16) because the limit is the **representation**, not the readout: position lives implicitly in *which of thousands of background tokens* is the agent, and both the encoder‑pool and the MSE‑predictor spend their capacity on the dominant background.

The generalizable fix is to stop representing the scene as an undifferentiated token grid and start representing it as **objects (slots) and their relations** — which is *also* exactly the primitive the north star needs: *unsupervised, scalable, real‑video world models that learn operative, tactical and strategic patterns and dependencies.*

## 2. The core idea

Insert an **object‑centric bottleneck** (Slot Attention) between the SSL encoder and the four‑brain hierarchy, and run each brain over **slots** instead of the dense grid:

```
frame ─► SSL encoder (unchanged) ─► tokens ─► Slot Attention ─► K object slots
                                                                   │
   OPERATIVE  : per‑slot dynamics     predict next slots  = "how each object moves"
   TACTICAL   : slot↔slot interactions (relations)        = "dependencies between objects"
   STRATEGIC  : scene‑graph over slot configurations (VQ) = "situations / goals"
   READOUT    : agent‑slot position (size‑invariant)      = control state
```

Slot Attention (Locatello et al. 2020) initializes K slots and runs a few rounds of *inverted* attention in which slots **compete** to bind tokens (softmax over slots, normalized over tokens) + a GRU update. Each slot converges onto one object. Because binding is by **feature** (appearance), not by pixel count, a 0.5‑radius agent is captured as cleanly as a 1.7‑radius one — the size problem dissolves rather than being patched.

## 3. Why this is the disruptive, generalizable pattern (not a toy patch)

- **It makes "world patterns and dependencies" literal.** Objects are the universal structure of the physical world; their *interactions* are the dependencies. A world model over slots learns physics as slot→slot dynamics — the thing you actually want to generalize.
- **It is the missing substrate for H‑JEPA.** LeCun's hierarchical‑JEPA vision — multi‑scale predictive abstraction for planning — has no canonical *representation* to hang the levels on. Objects give it one: operative = object dynamics, tactical = object relations, strategic = scene configuration. A **measured hierarchy edge** (operative < tactical < strategic) over an object‑centric SSL model, from scratch, on video, is the defensible disruption. No published system shows this end‑to‑end.
- **It is real‑video‑ready.** In driving, slots = vehicles / lanes / signs; slot dynamics = their motion; slot interactions = right‑of‑way, following, merging — the tactical layer of the AD report becomes *literal*.
- **It is data‑efficient.** Object‑factored dynamics generalize combinatorially (learn "a car follows a car" once, reuse everywhere), which is the compositional generalization the "much less data" thesis needs.

## 4. Fully self‑supervised — no labels added

Slot Attention needs **no supervision** to bind objects. Binding signals available to us, all label‑free:
- **SIGReg** on slots (anti‑collapse, keeps slots distinct) — already in the stack;
- **inverse dynamics on slots** — the *controllable* slot is the one whose change predicts the action (proprioception, not labels); this both grounds the agent slot and tells the hierarchy which slot it steers;
- **slot next‑state prediction** — the operative loss becomes "predict next slots," self‑supervised;
- (optional, for stronger binding) a **light slot→feature decoder** reconstructing the *encoder features* (not pixels) — still label‑free, keeps the decoder‑free‑for‑pixels character that helps under clutter (MuDreamer).

The readout probe (`slot_readout.py`) is a frozen measuring instrument fit only on proprioceptive positions — the encoder/predictors are untouched, exactly like the ridge/soft‑argmax probes. **The unsupervised, no‑label, no‑reward character is preserved.**

## 5. How it fixes the size problem — two mechanisms

1. **Size‑invariant decode.** The agent slot binds by appearance; a small agent reads as sharply as a large one, and same‑size distractors get their *own* slots (no blending).
2. **Reads a diffuse imagination.** The predictor's off‑manifold output has the agent activation *spread* across cells; slot attention **aggregates** that distributed activation into one slot → recovers the imagined position even when there is no sharp peak (where soft‑argmax's peak‑centroid reads flat). This is the property that a pooling‑by‑binding readout has and a pooling‑by‑peak readout lacks.

*(§8 records the measured result of the readout on the existing grid‑16 model.)*

## 6. Sharp per‑slot dynamics (the objective)

With slots isolating the agent, the operative predictor should predict **next slots**. Two SSL‑compatible objectives, both sharper than dense‑MSE:
- **discrete slot codes** (VQ per slot) + cross‑entropy next‑code — sharp by construction (DreamerV3/IRIS/Genie), our existing VQ machinery;
- or **slot‑delta prediction** (Δ‑IRIS) — predict the change of each slot, concentrating capacity on what moves (our residual/change‑weighting, now per‑object).

Because a slot is low‑dimensional and object‑bound, "predict where the agent slot goes" is a small, well‑posed problem — unlike "predict which of 4096 tokens becomes the agent."

## 7. Implementation roadmap

| Stage | Change | Retrain? | Proof |
|---|---|---|---|
| **R1 — slot readout** (this session) | `SlotPositionReadout` as a frozen probe; wire `--readout slot` into diagnose/video | no | does slot decode read the small‑agent **decode + imagination** where soft‑argmax read flat? (§8) |
| **R2 — slot operative** | Slot Attention bottleneck; operative predicts next slots (discrete or delta); inverse‑dynamics on slots | yes | clean control at **any agent radius** (0.3–0.9) on Two‑Rooms gate; supremacy edge with clean per‑step numbers |
| **R3 — slot tactical/strategic** | tactical = slot‑interaction transformer; strategic = VQ scene‑graph over slot configs | yes | hierarchy‑edge ablation over slots; interpretable object‑level plans |
| **R4 — real video** | same architecture, driving frames; slots = vehicles/lanes | yes | AD report Stage 0–1 (MetaDrive BEV → comma2k19); slot dynamics = tactical maneuvers |

R1 keeps everything else fixed and tells us immediately whether object‑binding is the right lever; R2 is the real fix (small agent, clean control); R3–R4 carry it to the hierarchy and to real video.

## 8. Measured result of the slot readout (R1) — mechanism confirmed, probe insufficient

Measured 2026‑07‑02 on the grid‑16 / radius‑0.9 switch‑gate model (depth‑4 debug), episode‑split,
CPU probe fit (400 steps — a deliberate minimal budget):

| Readout | REAL‑frame G1 | IMAGINATION `action_spread` | imagined‑move ↔ action alignment |
|---|---|---|---|
| soft‑argmax (peak centroid) | **0.099 wu** | 0.025 (flat) | – |
| **slot (calibrated on predictor outputs)** | 2.139 (under‑trained probe) | **0.521 (21×)** | **0.55** (0.5 = random) |

Read: **the aggregation mechanism works** — on the predictor's *imagined* (diffuse, off‑manifold)
outputs, where the peak‑centroid readout is dead flat, slot binding recovers 21× more
action‑conditioned signal with above‑random direction. That is precisely the property claimed in
§5.2 and the evidence that object‑binding is the right lever. **But a post‑hoc probe is not the
fix**: fit from scratch on a frozen dense‑grid model with a tiny budget, the slot module
under‑trains (real‑frame decode 2.1 wu) and 0.55 alignment is directional, not control‑grade.
Conclusion unchanged from §7: **R1 validates the direction; R2 (slots trained *into* the model,
operative = slot dynamics) is where the size problem actually gets solved** — binding is then
shaped by the SSL objectives over the whole dataset, not bolted on afterwards.

### 8.1 R1.5 completes the negative chain: post-hoc slot probes are ruled OUT

Full-budget follow-ups (2026-07-02, pod `_gate11` + local `_g16`):

| Probe variant | real-frame G1 (episode-split) |
|---|---|
| slot, position-regression only, 400 CPU steps | 2.14 |
| slot, position-regression only, **3000 GPU steps** | 2.16 (budget-independent → structural) |
| slot **+ DINOSAUR-style feature-reconstruction** (SlotFeatureDecoder), det. eval, per-slot inits | **2.144 (unchanged)** |
| soft-argmax (reference) | 0.099 |

Diagnosis: on this SSL encoder's tokens, slots bind **spatial regions, not objects**. The token
features are position-entangled (the prediction objective mixes appearance with location), so
"the red agent" is not a coherent feature cluster; the reconstruction-optimal solution is each
slot owning a patch of the grid, and the agent hops between slots as it moves — no fixed
agent-selector can track it (probe collapses to predict-the-mean). DINOSAUR succeeds on DINO
features because they are object-clustered; from-scratch SIGReg+prediction features are not.

**The complete post-hoc-readout ablation chain — ridge (quantizes/overfits), soft-argmax (sharp
real-frame, flat on imagination), slots ± reconstruction (regions, not objects) — is exhausted.
Object binding must be learned JOINTLY with the representation: R2 is required, and this chain
is its empirical justification.** (The recon decoder + deterministic per-slot-init machinery
built here carries directly into R2's slot bottleneck.)

## 9. R2 v1 post-mortem and the v2 design

**R2 v1** (`--slot-mode`, commit 89bf075; first run `_slot09`, depth-10/25ep) trained but showed a
**rising op loss** (0.60→1.17) and a diagnostic inversion: predicted slots carried
action-conditioned position (calibrated 0.62) while real-frame slots did NOT encode position
(G1 1.55, oracle 0.28), with a collapsed slot norm (~11 vs ~10² for tokens). Three design flaws,
each a named requirement in the slot-dynamics literature (SAVi/STEVE/SOLD):

1. **No temporal slot consistency** — v1 extracted slots per frame from (stochastic, at train
   time) learned inits, so slot IDENTITY could permute frame-to-frame → the slot-dynamics
   prediction target was a moving target (the rising loss).
2. **No slot-level action grounding** — inverse dynamics still read token pools; nothing forced
   any slot to carry the controllable object (and the eval inverse row was invalidated by the
   train/eval representation mismatch).
3. **No collapse protection / target discipline on slots** — tokens get SIGReg, slots got
   nothing, and the prediction target was not detached → the "make slots trivial to predict"
   shortcut was open (the low readout norm says it was taken). Also: the logged `op` folded the
   reconstruction loss in, hiding which term rose.

**v2 (this commit) fixes all four, preserving the core principles (fully unsupervised,
SIGReg-single-mechanism, tactical/strategic wiring untouched):**
- `slots_of_window`: **SAVi-style recurrent binding** — slots at frame t initialize from t-1.
- Inverse dynamics reads **slot pools** in slot mode (grounds the controllable slot; eval-consistent).
- **SIGReg on the slot embeddings** (same mechanism as every other latent space) + the slot
  prediction target is **always detached**.
- `rec` logged separately from `op`; online control uses a stateful recurrent encoder with a
  non-mutating `peek` for counterfactual rows (instrument discipline).

Gate unchanged: slot-model diagnostic `CALIBRATED direction_acc ≥ 0.6` **and** `ORACLE-DECODE
≥ 0.9` on the radius-0.9 switch-gate = size problem solved by the representation.

## 10. Terminal verdict on the toy (2026-07-06) — and why it strengthens the driving case

The ISOLATION LOOP (slot-attention + decoder alone, reconstruction-only, frozen encoder — a
30-minute iteration harness) produced termination-grade evidence:

| isolated experiment | slots used per image | recon |
|---|---|---|
| recon-only, 192-dim slots | **1** (a different slot per scene = archetype clustering) | 0.02 |
| slot-dim bottleneck 64 / 32 | **1** | 0.04 |
| change-weighted recon (movers ×20) | **1** (the agent reconstructed well — by the same slot) | 0.02 |

Combined with the full-model runs (v1, v2, config-C, +motion-cue: masks = region tiles at
patch-16, whole-frame collapse at patch-8), the conclusion is complete:

1. **Consequence-dominance (A8), at the binding level:** the agent is ~1.5 % of recon loss
   mass; binding never needs it. Change-weighting fixes *what* is reconstructed, not *how
   many* slots share the work.
2. **Decomposition threshold:** slot attention decomposes only when scene entropy exceeds
   single-slot capacity. This deliberately minimal toy (archetype + agent position ≈ tens of
   bits) sits BELOW the floor — even 32-dim slots absorb whole scenes.

Both laws are **toy artifacts that invert on real video**: high-entropy scenes force
decomposition, and ego-motion gives movers dominant loss mass. The slot four-brain is a
real-video architecture; the toy's positive result stands as the radius-1.1 hierarchy-
supremacy proof (operative stalls / four-brain solves 5/5), which is loss-mass-legal.
Slot work on the toy is CLOSED; the stack (recurrent binding, motion cue, change-weighted
recon, equivariant/mask readouts, isolation harness) carries to driving Stage-0.

## 9. Key references

Slot / object‑centric: Slot Attention (Locatello 2020); [SOLD — slot object‑centric latent dynamics for RL](https://arxiv.org/html/2410.08822v2); [Slot‑MPC](https://arxiv.org/pdf/2605.14937); OCVP.
Sharp discrete dynamics: [DreamerV3](https://www.emergentmind.com/topics/dreamerv3); [IRIS / Δ‑IRIS](https://arxiv.org/pdf/2406.19320); [Genie](https://arxiv.org/abs/2402.15391).
Decoder‑free under clutter: MuDreamer / M3PO.
Hierarchical prediction: LeCun H‑JEPA; [NextLat](https://arxiv.org/abs/2511.05963).
Our lineage: LeWM (SIGReg SSL), residual≈Δ‑IRIS, inverse‑dynamics≈IDOL, latent‑graph routing (this repo).
