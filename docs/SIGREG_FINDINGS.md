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
