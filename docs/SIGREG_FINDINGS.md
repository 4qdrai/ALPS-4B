# SIGReg-only collapse: investigation & findings (2026-06)

Goal: keep collapse prevention **SIGReg-only** (LeWorldModel / LeJEPA), no EMA, no
stop-gradient, no VICReg, no DINO-style teacher. This note records what the A40 run
revealed, the bugs fixed, and the diagnosis of the remaining pure-SSL collapse.

## What the A40 run showed (G1 identifiability gate)
- **SUP** (light position/dynamics anchor): G1 **0.232 wu**, eff-rank 16.6/192 — healthy.
- **SSL** (pure self-supervised, `--self-supervised`): G1 **3.513 wu**, eff-rank
  **2.3/192**, 168 dead dims, pairwise-cos **1.00** → **collapse**.
- The hierarchy edge itself is solid on the (grounded) model: ablation ladder
  cross-room **0.04 → 0.61**, oracle 0.71.

## Two genuine faithfulness bugs found & fixed
1. **SIGReg was effectively disabled.** `train_temporal.py` divided the loss by
   `n_rows = B*W*N`. The Epps-Pulley statistic already carries its `*N` factor (it is
   a test statistic), so dividing cancelled it and shrank SIGReg ~1000×. Reported
   `sig` was ~0.005 vs `op`≈1.0. **Fix:** apply `λ·SIGReg(Z)` directly (paper λ=0.1),
   on the per-frame pooled embedding.
2. **A stop-gradient LeWM explicitly forbids.** `L_op` used `z[1:].detach()`
   (BYOL-style), which forces reliance on extra collapse prevention. **Fix:** removed
   the detach (kept behind a diagnostic `--stopgrad-target` flag, default OFF).
3. Removed the bolted-on **VICReg** (`var_cov_reg`) entirely — SIGReg is the sole
   mechanism now. Also added **fresh random projections every step** in `sigreg.py`
   (LeJEPA resamples; a fixed projection set can be fooled by anisotropic collapse).

## Our SIGReg == official LeJEPA (verified)
Cloned `github.com/rbalestr-lab/lejepa`. Our `SIGReg.forward` is byte-for-byte the
official Epps-Pulley: same `knots=17`, `t∈[0,3]`, `weights`, `window=exp(-t²/2)`,
`statistic=(err@weights)*N`, no standardization, resampled directions. **Not a bug.**

## Remaining result: pure SSL still collapses on Two-Rooms-SIMPLE
Local sweep (d128, 1000-episode data ≈10k windows, frozen-probe eval):

| config | eff-rank | dead | cos | G1 |
|---|---|---|---|---|
| SSL λ=0.1, resampled, no-stopgrad | 1.2/128 | 124 | 1.00 | 3.55 |
| SSL λ=0.1, **fixed** projections   | 1.2/128 | 105 | 1.00 | 3.53 |
| SSL λ=0.1, **stop-grad** on        | 1.0/128 | 113 | 1.00 | 3.55 |
| SSL **λ=2.0** (strong)             | 1.0/128 | 128 | 1.00 | 3.55 (diverged, sig=38) |
| **SUP (light anchor)**             | **16.6/192** | 39 | 0.86 | **0.23** |

Every pure-SSL config collapses; the only healthy latent is the position-anchored one.

## Diagnosis (leading hypotheses, in order)
1. **Missing projector decoupling.** LeJEPA applies SIGReg + prediction to a
   *projection-head* output, and probes/decodes the *backbone* embedding (detached
   probe). We apply SIGReg directly to the decoded representation — over-constraining
   the very vector we read out. This is the standard SSL projector trick (SimCLR/
   VICReg/BYOL/LeJEPA all use it) and is the most likely structural cause.
2. **Task too trivial.** Two-Rooms-simple dynamics are near-identity (a dot drifts
   ~1 unit/step), so next-embedding prediction provides little pressure to encode
   position; SIGReg keeps the marginal Gaussian-ish but cannot inject task content.
   LeWM validated on richer 2D/3D control; LeJEPA on strongly-augmented ImageNette.

## Next experiments (no EMA, still SIGReg-only)
A. **Add a projection head** (encoder→backbone z used for decode/probe; small MLP
   projector p(z); apply prediction L_op and SIGReg on p(z); decode G1 from z). One
   architecture change; re-run the G1 SSL-vs-SUP gate.
B. **Richer prediction task**: longer operative horizon, the complex 4-room+key
   (key-state transitions force richer encoding), and especially **real video (H12)**
   where natural dynamics demand state. Pure SSL is expected to shine here.
C. **λ as the single gate** (LeJEPA's one hyperparameter), swept on the A40 — but
   λ∈{0.1,2.0} already bracket-failed locally, so A/B are higher-leverage.

Until A/B land, the hierarchy/edge results stand on the **position-anchored** encoder
(still frozen-probe eval); the pure-SSL claim is explicitly *open* and tied to the
projector + richer-task experiments above.

## Update — projector / BatchNorm investigated (2026-06)
LeWM §4 explicitly: the encoder appends a **1-layer MLP + BatchNorm** because "the
final ViT layer applies LayerNorm, which prevents our anti-collapse objective from
being optimized effectively," and "the predictor is also followed by a projector."
Our `encoders.py` already had a BN projection head — but with a **trailing GELU**,
which rectifies the embedding (non-negative) so it can never be a zero-mean isotropic
Gaussian. **Removed the GELU** (BN output is now the embedding, faithful to LeWM).

Result: removing the GELU did **not** resolve the collapse on Two-Rooms-simple
(d128, eff-rank 1.1, cos 1.00, G1 3.1). The remaining LeWM structural gap is the
**embedding granularity**: LeWM SIGRegs/decodes a **per-frame** BN embedding `z_t`,
whereas we BN **per-token** then mean-pool — pooling after per-token BN re-introduces
the cross-frame collapse the BN never constrained. Faithful next step is to pool the
ViT tokens to one per-frame vector, then MLP+BN → `z_t`, and SIGReg/decode that.

Combined with the trivial Two-Rooms dynamics (B), pure SSL is not the right vehicle
on the simple task. **Decision (per project owner): keep the SIGReg-only/no-EMA fixes,
treat pure-SSL identifiability as open (pursue per-frame embedding + richer task /
real video), and proceed NOW on the position-anchored encoder to train & validate the
strategic/tactical abstraction layers (latent prediction + goal emission), which is
the priority.** Those layers train on detached encoder features, so a healthy
(anchored) encoder is sufficient and necessary for them.

## Update 2 — abstraction layers + the BatchNorm eval gap (2026-06)
New gate `evaluation/validate_abstraction.py` measures the non-negotiable: do the
strategic/tactical layers (a) predict their own latent forward, (b) emit goals down
the hierarchy? Findings on a small anchored model (d128, ~24 ep, frozen-probe eval):

- **Goal emission WORKS**: `G_str2tac` PASS (conditioning the tactical predictor on a
  strategic goal-concept pulls its emitted region toward that goal's location) and
  `G_tac2op` PASS (the emitted sub-goal is an in-bounds operative target).
- **Latent prediction is learned but had a BatchNorm train/eval gap.** The encoder's
  BN projection head (needed for SIGReg and to keep the abstraction latents from
  collapsing) used running stats, so eval-mode features were shifted and the
  predictors — trained on train-mode features — broke (tactical predict-vs-standstill
  rel-err **6.26** at eval vs **0.72** in train mode). Fix: `BatchNorm1d(track_running_stats=False)`
  → batch stats at train AND eval. This also revived VQ diversity (codes **2 → 18**)
  and pulled the tactical eval rel-err **6.26 → 2.24 → 1.26** (approaching the train
  0.72; remaining gap is batch-composition + undertraining, expected to clear at
  A40 scale). Removing the BN head entirely is NOT an option — the abstraction latents
  then collapse (codes → 1). The gate is wired into the A40 script (stage 8d).

Net: the strategic/tactical layers DO predict in latent space and DO emit goals
downstream; the residual is closing the BN eval-consistency gap at scale.

## Update 3 — CORRECTION: LeWM uses THIS exact Two-Rooms task; collapse is a config gap, not triviality (2026-06)
Earlier hypothesis "B" (Two-Rooms too trivial for pure SSL) is **WRONG**. The LeWM
paper (§Environments) evaluates on **Two-Room** (Sobal et al.): "two rooms separated
by a wall with a single door... agent (red dot) navigates from a random start in one
room to a random target in the other, requiring passing through the door. We collect
**10,000 episodes**." That is *our* task. LeWM trains **pure SIGReg-only SSL (no
EMA/stop-grad)** on it and it does NOT collapse — and they probe physical quantities
(position) from the frozen latent. So pure-SSL identifiability is ACHIEVABLE here; our
collapse is an implementation/config gap vs LeWM, to close by matching their setup:

1. **SIGReg on the PREDICTOR outputs too**, not only encoder embeddings. LeWM applies
   SIGReg to z_t, z_{t+1}, AND the prediction ẑ_{t+1}, and "the predictor is also
   followed by a projector network with the same implementation as the encoder"
   (i.e., its own Linear+BatchNorm). We currently SIGReg only the encoder embedding.
2. **Data scale**: LeWM uses 10,000 episodes; our local tests used 1-3k. Match 10k.
3. **Inter-frame motion**: prediction must be non-trivial — ensure the sub-trajectory
   frame spacing produces meaningful agent displacement (so next-embedding prediction
   requires encoding position). Tune `stride`/`frame-skip` to LeWM's regime.
4. Architecture parity: ViT-Tiny, patch 14, BatchNorm projector on BOTH encoder and
   predictor (we have the encoder one; ADD the predictor projector).

This is the SSL path (no EMA). Until it lands, the anchored hierarchy (default) gives
the healthy encoder the strategic/tactical layers need.
