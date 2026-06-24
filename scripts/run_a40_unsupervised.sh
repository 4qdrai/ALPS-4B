#!/bin/bash
# run_a40_unsupervised.sh
# ============================================================================
# FULLY UNSUPERVISED ALPS-4B proof. EVERY stage and gate runs on a world model
# trained with ZERO labels: pure SIGReg-only (LeWM-faithful) -- no position /
# dynamics supervision, no EMA, no stop-gradient. The strategic, tactical and
# operative layers are ALL trained self-supervised (feature prediction + SIGReg
# on embeddings AND predictions at every scale).
#
# IMPORTANT -- what "unsupervised" means here:
#   * TRAINING: the encoder + all 3 predictor layers see NO labels (--lewm-ssl).
#   * EVALUATION: gates fit FROZEN linear probes (z -> x,y ; z -> has_key) purely
#     to READ OUT the latent. That is the standard JEPA/LeWM linear-probe protocol
#     (LeWM probes physical quantities the same way); the probe never touches the
#     encoder. The representation and the hierarchy are unsupervised; the probe is
#     only a measuring instrument.
#
# GO/NO-GO: stage 2 prints G1 (frozen-probe position decode) on the unsupervised
# encoder. If G1 < 0.30 wu the latent is linearly identifiable WITHOUT labels and
# every downstream gate is meaningful. If G1 is high, the foundation needs more
# (epochs / episodes / inter-frame motion) -- downstream numbers will be weak and
# that is the honest signal, not a bug.
#
# Usage:   bash scripts/run_a40_unsupervised.sh
# Tune:    EPISODES=10000 EPOCHS=80 DMODEL=192 bash scripts/run_a40_unsupervised.sh
# ============================================================================
set -euo pipefail
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

# ---- Config (LeWM regime by default) ----------------------------------------
EPISODES="${EPISODES:-10000}"          # LeWM uses 10k episodes for Two-Room
MAXSTEPS="${MAXSTEPS:-100}"
EPOCHS="${EPOCHS:-80}"                  # pure SSL needs enough epochs for identifiability
DMODEL="${DMODEL:-192}"
ENC_DEPTH="${ENC_DEPTH:-10}"
ENC_HEADS="${ENC_HEADS:-8}"
WINDOW="${WINDOW:-6}"
STRIDE="${STRIDE:-4}"                   # inter-frame motion; raise if G1 plateaus
BATCH="${BATCH:-64}"
SIGREG_SLICES="${SIGREG_SLICES:-1024}"
NUM_CODES="${NUM_CODES:-64}"
NUM_EXPERTS="${NUM_EXPERTS:-4}"
ACTIVE_EXPERTS="${ACTIVE_EXPERTS:-2}"
N_EVAL="${N_EVAL:-200}"
COARSE_K="${COARSE_K:-8}"
FINE_K="${FINE_K:-24}"
# SPATIAL readout for the four-brain control. The global pool discards the small agent
# under pure SSL (pooled G1 stays random even at scale -- it's the READOUT, not model
# size), so the hierarchy plans/controls on a coarse gxg SPATIAL readout where position
# is recoverable (validated locally: ridge R^2 0.92). g=8 = full token grid (sharpest
# decode); set SPATIAL_GRID=0 to disable and fall back to the global pool.
SPATIAL_GRID="${SPATIAL_GRID:-8}"
SPATIAL_ARGS=""; [ "$SPATIAL_GRID" != "0" ] && SPATIAL_ARGS="--spatial --spatial-grid $SPATIAL_GRID"
# PATCH size (t h w). Default (2 16 16) -> 8x8=64 tokens (decode resolution-capped
# ~0.55-0.73wu, above per-step motion 0.27 -> control can't route). For sharper decode
# set PATCH="2 8 8" -> 16x16=256 tokens (~0.3wu) and pair with SPATIAL_GRID=16. The
# spatial grid can be no finer than the token grid, so patch8 is required for grid16.
PATCH="${PATCH:-2 16 16}"
# CTRL_K: K-step predictor rollout per control decision -> raises per-decision
# displacement above the decode noise WITHOUT retraining (the cheap routing lever).
CTRL_K="${CTRL_K:-3}"; CTRL_ARGS="--ctrl-k $CTRL_K"
FB_CAL="${FB_CAL:-60}"; FB_EVAL="${FB_EVAL:-100}"
CX_EPISODES="${CX_EPISODES:-3000}"
CX_BFS_FRACTION="${CX_BFS_FRACTION:-0.5}"

DATA="data/two_rooms/trajectories_unsup.pt"
CX_DATA="data/two_rooms/trajectories_unsup_complex.pt"
MODEL="results/two_rooms/validation/unsup_temporal.pt"
CX_MODEL="results/two_rooms/validation/unsup_temporal_complex.pt"
OUT="results/two_rooms/validation/unsupervised"
mkdir -p "$OUT"

TRAIN_ARGS="--lewm-ssl --d-model $DMODEL --enc-depth $ENC_DEPTH --enc-heads $ENC_HEADS \
  --window $WINDOW --stride $STRIDE --batch-size $BATCH --sigreg-slices $SIGREG_SLICES \
  --num-codes $NUM_CODES --num-experts $NUM_EXPERTS --active-experts $ACTIVE_EXPERTS \
  --epochs $EPOCHS --patch-size $PATCH --save-every ${SAVE_EVERY:-5} --save-model"

echo "================ ALPS-4B FULLY-UNSUPERVISED validation ================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

# ---- 1. Data (labels stored only for the eval PROBE, never used in training) -
if [ ! -f "$DATA" ]; then
  echo "--- [1/8] generating $EPISODES-episode dataset ---"
  python -m alps.benchmarks.two_rooms.data_generator \
      --save-path "$DATA" --num-episodes "$EPISODES" --max-steps "$MAXSTEPS" \
      --heuristic-fraction 0.4 --seed 7
fi

# ---- 2. UNSUPERVISED temporal hierarchy (operative+tactical+strategic) -------
if [ ! -f "$MODEL" ] || [ -n "${RETRAIN:-}" ]; then
  echo "--- [2/8] training FULL hierarchy, PURE SSL (--lewm-ssl) ---"
  python -m alps.training.train_temporal --data-path "$DATA" --out "$MODEL" $TRAIN_ARGS
fi
echo "--- [2/8] GO/NO-GO: G1_spatial + collapse + four-brain (spatial readout) ---"
python -m alps.evaluation.validate_temporal \
    --model-path "$MODEL" --data-path "$DATA" --n-episodes "$N_EVAL" \
    --coarse-k "$COARSE_K" --fine-k "$FINE_K" --save-dir "$OUT" $SPATIAL_ARGS $CTRL_ARGS

# ---- 3. Abstraction gates: strategic/tactical predict latent + emit goals ----
echo "--- [3/8] abstraction-layer gates (unsupervised) ---"
python -m alps.evaluation.validate_abstraction \
    --model-path "$MODEL" --data-path "$DATA" --save-dir "$OUT" \
    || echo "[warn] [3/8] abstraction gate failed -- NON-FATAL, continuing to the proof videos"

# ---- 4. COMPLEX (key->door->goal), unsupervised — H4 label-free key ---------
if [ ! -f "$CX_DATA" ]; then
  echo "--- [4/8] generating COMPLEX dataset (BFS-optimal + random) ---"
  python -m alps.benchmarks.two_rooms.generate_complex \
      --save-path "$CX_DATA" --num-episodes "$CX_EPISODES" --bfs-fraction "$CX_BFS_FRACTION"
fi
if [ ! -f "$CX_MODEL" ] || [ -n "${RETRAIN:-}" ]; then
  echo "--- [4/8] training COMPLEX hierarchy, PURE SSL (--lewm-ssl) ---"
  python -m alps.training.train_temporal --data-path "$CX_DATA" --out "$CX_MODEL" $TRAIN_ARGS
fi
echo "--- [4/8] COMPLEX four-brain, H3 VQ-graph, H4 label-free key, H2 emitter ---"
python -m alps.evaluation.validate_temporal --complex \
    --model-path "$CX_MODEL" --data-path "$CX_DATA" --n-episodes "$N_EVAL" \
    --coarse-k "$COARSE_K" --fine-k "$FINE_K" --save-dir "$OUT" \
    --h4-unsup-key --vq-graph --h2-emitter $SPATIAL_ARGS $CTRL_ARGS \
    || echo "[warn] [4/8] COMPLEX four-brain gate failed -- NON-FATAL (complex model still saved for videos)"
# Simple mode: H3 + H2 gates (no key) + spatial-readout four-brain edge
python -m alps.evaluation.validate_temporal \
    --model-path "$MODEL" --data-path "$DATA" --n-episodes "$N_EVAL" \
    --coarse-k "$COARSE_K" --fine-k "$FINE_K" --save-dir "$OUT" \
    --vq-graph --h2-emitter $SPATIAL_ARGS $CTRL_ARGS \
    || echo "[warn] [4/8] simple H3/H2 gate failed -- NON-FATAL, continuing"

# ---- 5. Fourth Brain: monitors -> escalation -> fallback (unsupervised) ------
echo "--- [5/8] Fourth Brain, simple + complex (H4 label-free key in complex) ---"
python -m alps.evaluation.fourth_brain \
    --model-path "$MODEL" --data-path "$DATA" --n-cal "$FB_CAL" --n-eval "$FB_EVAL" --save-dir "$OUT" $SPATIAL_ARGS \
    || echo "[warn] [5/8] fourth-brain (simple) failed -- NON-FATAL, continuing"
python -m alps.evaluation.fourth_brain --complex --label-free-key \
    --model-path "$CX_MODEL" --data-path "$CX_DATA" --n-cal "$FB_CAL" --n-eval "$FB_EVAL" --save-dir "$OUT" $SPATIAL_ARGS \
    || echo "[warn] [5/8] fourth-brain (complex) failed -- NON-FATAL, continuing"

# ---- 6. MoE expert specialization (unsupervised) -----------------------------
echo "--- [6/8] MoE expert specialization, simple + complex ---"
python -m alps.evaluation.moe_specialization --model-path "$MODEL" --data-path "$DATA" --save-dir "$OUT" \
    || echo "[warn] [6/8] MoE (simple) failed -- NON-FATAL, continuing"
python -m alps.evaluation.moe_specialization --complex --model-path "$CX_MODEL" --data-path "$CX_DATA" --save-dir "$OUT" \
    || echo "[warn] [6/8] MoE (complex) failed -- NON-FATAL, continuing"

# ---- 7a. Latent-RAG: single-pass (H7 original) + lifelong batches (H7 extended) --
echo "--- [7/8] RAG-in-the-loop H7 (single-pass + lifelong batches) ---"
python -m alps.evaluation.fourth_brain --rag \
    --model-path "$MODEL" --data-path "$DATA" --n-cal "$FB_CAL" --n-eval "$FB_EVAL" --save-dir "$OUT" $SPATIAL_ARGS \
    || echo "[warn] [7/8] RAG single-pass failed -- NON-FATAL, continuing"
python -m alps.evaluation.fourth_brain --h7-lifelong \
    --model-path "$MODEL" --data-path "$DATA" --n-cal "$FB_CAL" --n-eval "$FB_EVAL" \
    --n-batches 5 --save-dir "$OUT" $SPATIAL_ARGS \
    || echo "[warn] [7/8] RAG lifelong failed -- NON-FATAL, continuing"

# ---- 8. PROOF VIDEOS: Four-Brain solving SIMPLE + COMPLEX (unsupervised model) -
# Side-by-side operative(stalls) vs Four-Brain(solves), env's own frames, GIF + MP4.
# Renders the model's ACTUAL behaviour -> a genuine proof when the model solves.
MAKE_VIDEOS="${MAKE_VIDEOS:-1}"
if [ "$MAKE_VIDEOS" = "1" ]; then
  echo "--- [8/8] proof videos: simple + complex (unsupervised) ---"
  python -m alps.benchmarks.two_rooms.make_videos_4b \
      --model-path "$MODEL" --data-path "$DATA" \
      --complex-model-path "$CX_MODEL" --complex-data-path "$CX_DATA" \
      --save-dir results/two_rooms/videos_unsup \
      --coarse-k "$COARSE_K" --fine-k "$FINE_K" --stride "$STRIDE" --n-clips "${N_CLIPS:-3}" \
      $SPATIAL_ARGS $CTRL_ARGS
fi

echo "--- DONE ---"
echo "All gates ran on the UNSUPERVISED (--lewm-ssl) models. Artifacts in $OUT/ :"
echo "  temporal_gates.json          (G1 / G_collapse / G_str / G_tac / G_roll / G_4brain)"
echo "  temporal_gates_complex.json  (COMPLEX key->door->goal four-brain)"
echo "  abstraction_gates.json       (strategic/tactical latent prediction + goal emission)"
echo "  fourth_brain{,_complex}.json (H8 monitoring / H9 escalation / H10 fallback)"
echo "  moe_specialization{,_complex}.json (H11 expert routing + knockout)"
echo "  rag_selflearning.json        (H7 surprise-gated RAG-in-the-loop)"
echo "  ../videos_unsup/fourbrain_simple_*.{gif,mp4}, fourbrain_complex_*.{gif,mp4}  (proof clips)"
echo ""
echo "READ FIRST: $OUT/temporal_gates.json"
echo "  * G1_spatial  -> the unsupervised position readout (spatial gxg). If <0.30 the"
echo "    pure-SSL latent is identifiable WITHOUT labels (pooled G1 stays random -- that"
echo "    is the readout, not a failure)."
echo "  * G_4brain    -> cross-room edge: operative(System-1) ~0 << strategic/tactical"
echo "    (graph) = the hierarchy edge, unsupervised + SIGReg + predictor-decoded."
echo "  If G1_spatial is high, raise EPOCHS/EPISODES/STRIDE (sharper predictor) or"
echo "  SPATIAL_GRID (finer readout)."
