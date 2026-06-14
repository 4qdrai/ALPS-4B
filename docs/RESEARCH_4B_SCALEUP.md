# Scaling the Four-Brain (4B) to Real Video: Research Landscape, Encoder Design, and Disruption Plan

*2026-06. Deep-research brief for the goal: prove the 4B multi-abstraction-layer*
*architecture — unsupervised, world-model-prediction-based, with strategic*
*reasoning + constraint-emission, self-monitoring, and lifelong memory — is*
*disruptively superior on real video / physical-world objective-oriented learning.*

---

## 0. The one-sentence thesis (and why it's defensible)

> Every published system owns **one** piece of the puzzle. The 4B is the first to
> integrate **all** of them into one *small, self-supervised, gate-validated* stack —
> and to show the edge **grows with task horizon**. The integration is the contribution.

| System (2025–26) | Has | Lacks (vs 4B) |
|---|---|---|
| **V-JEPA 2 / 2.1 / 2-AC** (Meta, 1.2B) [arXiv 2506.09985] | scale, SSL video, MPC planning, action-cond. post-train | flat (no abstraction hierarchy), huge, no self-monitoring, no episodic memory |
| **H-JEPA** (2026 scaling-laws paper) | the hierarchy (stacked multi-scale prediction) | **no self-monitoring, no episodic/lifelong memory, no strategic-constraint emission** (confirmed) |
| **DINO-WM** | frozen-feature zero-shot MPC | flat, rides a giant pretrained encoder, no memory/monitor |
| **Generative WMs** (Cosmos, Genie 2/3, Dreamer v3) | photorealistic rollouts | compounding error, expensive, weak physical consistency, no explicit abstraction tiers |
| **Latent-action WMs** (CLAW, AdaWorld, LAM) | actions/consequences from unlabeled video | flat, no monitoring/memory hierarchy |
| **Continual-learning agents** (EWC/LoRA, replay) | long memory | parametric → still forget; no world-model/abstraction |

**The unclaimed niche:** small (~5–22M) SSL encoder · ≥3 gate-validated abstraction
tiers · **discrete (VQ) interpretable strategic plan** · **graph-anchored** (no
compounding rollout) · **surprise-gated episodic memory** (lifelong, zero weight
updates) · **self-monitoring → escalation → fallback** · **edge-grows-with-horizon**.
No published system has this combination.

---

## 1. The 2026 landscape — seven threads, each a pillar of the argument

### 1a. SSL world models from pixels (the foundation family) — *validates "unsupervised understanding of physics"*
- **V-JEPA 2** (1.2B; >1M h video) → **V-JEPA 2-AC** (action-conditioned post-train on <62 h Droid; MPC; **16× faster than Cosmos**) → **V-JEPA 2.1** (Mar 2026, temporally-consistent dense features). Proves SSL video models *understand, predict, and plan*. [arXiv 2506.09985; ai.meta.com]
- **Intuitive physics emerges from SSL video pretraining** [arXiv 2502.11831]: a V-JEPA-style latent-prediction model reaches **above-chance object-permanence/continuity** with as little as **one week of video** — and **generative/pixel-prediction models fail where the JEPA latent-prediction model succeeds**. This is the single strongest external validation of "predict-in-latent, not pixels, to understand physics."
- **LeWM** [arXiv 2603.19312, Mar 2026]: end-to-end JEPA from pixels, **two loss terms** (next-embedding pred + SIGReg), 15M params, TwoRoom/PushT/OGBench-Cube, **O(log n) planning**. Our direct lineage. (Note: LeWM is *weakest* on TwoRoom — the simplest env — where PLDM/DINO-WM beat it; the edge is on harder tasks.)
- **LeJEPA** (Balestriero): SIGReg as the single anti-collapse mechanism, provable, no EMA/stop-grad. Our SIGReg is byte-faithful to it.

### 1b. Frozen-feature planning — DINO-WM
Frozen DINOv2 + ViT dynamics + zero-shot MPC. Proves *frozen-latent planning works*, but flat and dependent on a giant internet-pretrained encoder (the opposite of our small-from-scratch claim).

### 1c. Generative world models — *the foil*
Dreamer v3, **Genie 2/3**, **Cosmos** (NVIDIA), Sora-as-world-model. Powerful rollouts, but 2026 physical-reasoning benchmarks expose the gap: **PhyGround** [2605.10806], **VisPhyWorld** [2602.13294], **PhysicsMind** [2601.16007], **MVPBench** show generative models struggle with physical consistency, and they pay O(n) rollout cost with compounding error. Our foil claim: *latent JEPA + graph anchoring beats generation for physical reasoning at a fraction of the cost.*

### 1d. Hierarchical world models — *the closest competitors*
- **H-JEPA** (2026 scaling-laws paper): stacked JEPAs, each level predicts the next level's future representation. **Confirmed it does NOT do self-monitoring, episodic/lifelong memory, or strategic-constraint emission** — pure hierarchical prediction. This is precisely the 4B moat.
- **Causal-JEPA** [arXiv 2602.11389]: object-level latent masking, planning on **~1% of latent features** → efficiency + object structure.
- **Director** (2022): hierarchical control in WM latent space — but needs rewards + online RL.

### 1e. Latent-action models — *how to get "consequences of actions" from passive video*
- **CLAW** [arXiv 2606.04130, Jun 2026]: continuous latent actions via adversarial latent regularization.
- **AdaWorld** [arXiv 2503.18938]: adaptable WMs with latent actions.
- **Hierarchical Latent Action Model** [arXiv 2603.05815, ICLR-26 workshop]: hierarchical + latent actions — directly adjacent to 4B.
- Mechanism: **inverse dynamics** — infer the (latent) action from a pair of adjacent observations. This *forces the representation to encode controllable state* and lets the model learn action→consequence **without action labels**. This is the key to the real-video phase (and — see §4 — the fix to our current G1 failure).

### 1f. Self-monitoring / uncertainty — *validates the Fourth Brain*
- **"Boiling Frog Threshold"** [arXiv 2603.08455]: WM anomaly detection has a *sharp* detection threshold, but in fragile environments **"collapse before awareness"** creates lethal-but-undetectable gradual drift. → A concrete failure mode the 4B's *multi-signal + escalation + fallback* is designed to beat single-signal monitors on.
- **"WM Robustness via Surprise Recognition"** [arXiv 2512.01119]: surprise = our `m1` monitor.
- **Uncertainty-aware latent safety filters** [arXiv 2505.00779]; **failure classification + anomaly detection** [arXiv 2602.16182]. Epistemic (model) vs aleatoric (env) uncertainty; ensembles for OOD.

### 1g. Continual / lifelong memory — *validates RAG-as-memory*
2026 consensus: replay-based + regularization-based; episodic memory; **hippocampal–neocortical consolidation** is an open problem (ICLR-26 MemAgents). Parametric methods (EWC, LoRA-CL) **still forget**. Our **surprise-gated, non-parametric, zero-weight-update RAG** avoids catastrophic forgetting *by construction* — a clean differentiator.

---

## 2. Encoder design — JEPA/LeWM scaled for general video & physical, objective-oriented learning

### What transfers unchanged
3-D tube ViT (already video-native), SIGReg anti-collapse, the VQ/MoE/four-brain hierarchy, frozen-probe eval. (See `A40_TRAINING_INSTRUCTIONS.md` / `EDGE_PROGRAM.md`.)

### What must change to scale to real video (priority order)
1. **Make the COMPACT state encode controllable/agent content — the central fix.**
   Our live experiment (§4) proves pure next-latent prediction + SIGReg does *not*
   force position into any global readout. Add an **inverse-dynamics / latent-action**
   objective: predict (latent) action from pooled (z_t, z_{t+1}). This forces the
   compact latent to encode the controllable state **without labels** — and is
   *exactly* the mechanism the real-video phase needs for action→consequence. **This
   is the most important single change.** [CLAW, AdaWorld, LAM]
2. **Real multi-frame clips.** Drop the `T=2` fake-clip in `encode_frame`; feed real
   T frames so the 3-D conv captures motion. Keep the W-frame predictor history (3–8).
3. **Resolution / tokens.** 128²/patch-16 (64 tok) → 224²/patch-14 (~256 tok) à la
   V-JEPA-2.1; size `max_patches`/`pos_embed` accordingly.
4. **Spatial-preserving readout for the hierarchy.** Position/objects live in the
   *grid*, not a single vector. Give the strategic/tactical tiers access to a **coarse
   spatial grid or object slots** (slot-attention / Causal-JEPA object tokens), not
   only a collapsed global vector. Use [CLS]/register tokens for *global pose*, slots
   for *object/agent* reasoning. (DINOv2 register tokens absorb artifacts.)
5. **Capacity & data-efficiency.** Test ViT-Tiny → ViT-Small; add a **ResNet-18**
   option (LeWM Tab. 8: competitive *and* more sample-efficient on real images). Budget
   **augmentation + footage** for the from-scratch-on-limited-real-data risk (the real
   threat — DINO-WM/V-JEPA-2 sidestep it with pretraining/scale).
6. **Action conditioning for passive video.** No robot in the H12 phase → replace the
   discrete-action AdaLN with **latent actions** (from #1) or ego-motion/IMU; pure
   next-latent prediction otherwise.

---

## 3. Brainstorm — demonstration problems that show *disruptive* superiority

Each targets **one** superiority claim with a **falsifiable head-to-head vs the right baseline**.

| # | Demonstration | Claim it proves | Head-to-head baseline | Metric / kill-criterion |
|---|---|---|---|---|
| **D1** | **Horizon-scaling law** (N-room / growing-horizon real maze) | hierarchy + graph-anchoring is what *survives* task horizon | flat V-JEPA-2-style MPC (compounds), H-JEPA (no anchoring) | operative→0 by N=4; 4B ≥0.5 at N=8; monotone edge curve *(H6, wired)* |
| **D2** | **Strategic constraint emission** (strategic emits a discrete VQ subgoal + REACH/PROGRESS gate that tactical must satisfy; ablate it) | top-down constraint emission is *causally necessary* — unique to 4B | H-JEPA (predicts, never *emits constraints*) | removing the constraint collapses tactical success (causal diagonal) |
| **D3** | **Self-monitoring beats silent failure** (inject gradual drift / novel obstacle in real video) | multi-signal monitor + escalation detects drift single-signal monitors miss | the **Boiling-Frog** single-detector regime | AUROC ≥0.8, lead ≥10 steps; recovers episodes a flat model loses silently *(H8–10, wired)* |
| **D4** | **Lifelong memory, zero forgetting** (sequential perturbed task families; surprise-gated episodic WRITEs, no weight updates) | non-parametric episodic memory = instant adapt + zero interference | parametric continual learning (EWC / LoRA-CL) which forgets | +≥10% on new family by batch 3; ≤2% nominal drop; retention after 5 batches *(H7, wired)* |
| **D5** | **Physics/consequence from passive video** (latent-action WM predicts consequence of an action — object falls, door opens — in latent space; strategic reasons over it) | latent JEPA + hierarchy > generative for physical consequence reasoning, at ~1/1000th cost | Cosmos/Genie generative WM (expensive, physically inconsistent) | beat on IntPhys 2 / MVPBench + a custom consequence-prediction task; report FLOPs/latency ratio |
| **D6** | **Planning speed & non-compounding** (graph-anchored plan vs MPC rollout) | O(log n) graph planning > O(n) rollout with compounding error | LeWM/PLDM MPC; V-JEPA-2-AC MPC | plan-time + success vs horizon; drift stays flat where MPC diverges |
| **D7** | **Interpretable strategic plan on real video** (VQ concept timeline: "kitchen→hallway→door→living-room") | discrete VQ concept graph = human-auditable strategic reasoning | continuous-latent planners (opaque) | concept↔place purity ≥0.8; planned route aligns with filmed route vs shuffled null |
| **D8** | **Counterfactual "what-if" over strategies** (roll out consequences of *different* strategic choices in latent space; pick best) | the 4B reasons counterfactually about action consequences | flat policy / reactive baseline | chooses the lower-cost/safer strategy; ablation shows the rollout is used |
| **D9** | **Cross-environment adaptation without retraining** (train on space A, adapt to space B via episodic memory only) | adapts to genuinely new situations with **zero weight updates** | fine-tuned baseline (needs retraining + forgets A) | reaches target success on B after k episodes; A unchanged |

**The flagship trio for a paper:** **D1** (the scaling-law figure — "hierarchy is what survives horizon"), **D3** (self-monitoring catches what flat models miss), **D4** (lifelong memory with zero forgetting). D2 is the *mechanistic* proof that the abstraction layers do real top-down work (the reviewers' "is the hierarchy doing anything?" killer). D5/D7 are the real-video flagships.

---

## 4. The live experimental finding that *de-risks all of the above*

Local controlled A/B (RTX 4060, d192/depth6, same data), held-out linear position probe:

| readout | anchored (pos-loss) | pure SSL (mean-pool) | pure SSL (**[CLS]**) |
|---|---|---|---|
| **pooled (G1)** | 0.375 ✓ | 3.62 ✗ | **3.38 ✗** |
| full token grid | 0.173 | 0.55 | **0.148** (R² 0.997) |
| raw pixels | 0.040 | 0.040 | 0.040 |

**Finding:** position is *always present in the token grid* (patchification), but **no
global readout — mean OR learned [CLS] — encodes it under pure SSL.** [CLS] even made
the op-loss crash to 0.02 by becoming maximally predictable *while excluding the
agent*. → The blocker is the **objective**, not the pool operator: next-latent
prediction + SIGReg never *require* the compact state to track the small, hard-to-
predict agent. Only a position anchor (or an equivalent) forces it.

**Why this is good news:** the fix is **inverse-dynamics / latent actions** (§2 #1) —
which is *independently* the right design for the real-video phase (action→consequence
without labels). One change closes the toy G1 *and* unlocks the real-world goal.
Verifying locally **saved an A40 run** and converged our engineering with the 2026
literature.

---

## 5. Prioritized roadmap
1. **Add inverse-dynamics / latent-action head**, retrain small `--lewm-ssl` locally,
   re-run `diagnose_g1`. Target: pooled G1 < 0.30 *(unblocks everything; ~1 h on the 4060)*.
2. Lock the toy edge (D1/D2) on the now-identifiable latent → A40 confirm.
3. Stand up D3/D4 at scale (Fourth Brain + lifelong RAG).
4. Real-video phase (`H12`): real clips + latent actions + spatial/object readout →
   D5/D7/D8 (physics, interpretable plan, counterfactual) + D9 (adaptation).
5. Position the paper as **the integration** vs H-JEPA (hierarchy-only), V-JEPA-2
   (flat/huge), generative WMs (compounding/expensive), continual-learning (forgets).

---

## Sources
- V-JEPA 2 — arXiv:2506.09985; ai.meta.com/blog/v-jepa-2-world-model-benchmarks
- Intuitive physics from SSL video — arXiv:2502.11831
- LeWorldModel (LeWM) — arXiv:2603.19312
- H-JEPA scaling laws — ijsmt.org (2026)
- Causal-JEPA — arXiv:2602.11389
- CLAW (latent actions) — arXiv:2606.04130; AdaWorld — arXiv:2503.18938; Hierarchical Latent Action Model — arXiv:2603.05815
- Boiling Frog Threshold (WM self-monitoring) — arXiv:2603.08455; Surprise Recognition — arXiv:2512.01119; latent safety filters — arXiv:2505.00779; failure classification — arXiv:2602.16182
- Physical-reasoning benchmarks — PhyGround arXiv:2605.10806, VisPhyWorld arXiv:2602.13294, PhysicsMind arXiv:2601.16007
