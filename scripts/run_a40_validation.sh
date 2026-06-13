#!/bin/bash
# run_a40_validation.sh
# End-to-end ALPS-4B Two Rooms validation program on an A40 (or any CUDA GPU).
# Runs: data generation -> G1/G2 gates -> planning ablation ladder -> self-learning.
# See docs/VALIDATION_PLAN.md for the rationale and acceptance thresholds.
#
# Usage:
#   bash scripts/run_a40_validation.sh
# Override defaults via env vars, e.g.:
#   EPISODES=5000 EPOCHS=120 DMODEL=256 bash scripts/run_a40_validation.sh

set -euo pipefail

export PYTHONPATH=src
export PYTHONUNBUFFERED=1

# ---- Config (override via env) ----------------------------------------------
EPISODES="${EPISODES:-3000}"      # number of trajectory episodes to generate
MAXSTEPS="${MAXSTEPS:-100}"
FRAME_SKIP="${FRAME_SKIP:-4}"     # consecutive clip frames differ by ~1.2 world units
DMODEL="${DMODEL:-192}"           # latent width (must be divisible by ENC_HEADS)
ENC_DEPTH="${ENC_DEPTH:-10}"      # encoder ViT depth (≈ViT-Tiny at d192,depth12; 4 = sub-tiny)
ENC_HEADS="${ENC_HEADS:-8}"       # encoder heads (must divide DMODEL: 8 works for 128/192/256)
EPOCHS="${EPOCHS:-100}"
POS_WEIGHT="${POS_WEIGHT:-1.0}"
DYN_WEIGHT="${DYN_WEIGHT:-1.0}"
SIGREG_SLICES="${SIGREG_SLICES:-512}"
PROBE_EPOCHS="${PROBE_EPOCHS:-150}"
N_EPISODES_EVAL="${N_EPISODES_EVAL:-200}"
GRAPH_K="${GRAPH_K:-24}"
COARSE_K="${COARSE_K:-8}"             # COARSE strategic landmarks for the Four-Brain ablation
NUM_CODES="${NUM_CODES:-64}"          # strategic VQ codebook size
NUM_EXPERTS="${NUM_EXPERTS:-4}"       # tactical MoE experts
ACTIVE_EXPERTS="${ACTIVE_EXPERTS:-2}"
DATA="data/two_rooms/trajectories.pt"
MODEL="results/two_rooms/validation/repr_world_model_fs4.pt"

echo "================ ALPS-4B A40 validation ================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

# ---- 1. Generate dataset (skip if present) ----------------------------------
if [ ! -f "$DATA" ]; then
  echo "--- [1/4] generating $EPISODES-episode dataset ---"
  python -m alps.benchmarks.two_rooms.data_generator \
      --save-path "$DATA" --num-episodes "$EPISODES" --max-steps "$MAXSTEPS" \
      --heuristic-fraction 0.4 --seed 7
else
  echo "--- [1/4] dataset already exists at $DATA (skipping) ---"
fi

# ---- 2. Train world model + run gates G1/G2 ---------------------------------
# Skip if already trained (set RETRAIN=1 to force). Lets you resume after a later
# stage failed without re-doing the ~1h foundation training.
if [ -f "$MODEL" ] && [ -z "${RETRAIN:-}" ]; then
  echo "--- [2/4] ReprWorldModel already exists at $MODEL (skipping; set RETRAIN=1 to force) ---"
else
  echo "--- [2/4] training ReprWorldModel + gates G1/G2 ---"
  python -m alps.evaluation.repr_decoder_gate train \
      --data-path "$DATA" --frame-skip "$FRAME_SKIP" --d-model "$DMODEL" \
      --enc-depth "$ENC_DEPTH" --enc-heads "$ENC_HEADS" \
      --epochs "$EPOCHS" --pos-weight "$POS_WEIGHT" --dyn-weight "$DYN_WEIGHT" \
      --sigreg-slices "$SIGREG_SLICES" --probe-epochs "$PROBE_EPOCHS" --save-model
fi

# ---- 3. Planning ablation ladder (the edge evidence) ------------------------
echo "--- [3/4] ablation ladder (rungs 0/2/4a/4b/5) ---"
python -m alps.benchmarks.two_rooms.run_ablation_ladder \
    --model-path "$MODEL" --data-path "$DATA" --frame-skip "$FRAME_SKIP" \
    --n-episodes "$N_EPISODES_EVAL" --graph-k "$GRAPH_K"

# ---- 4. Self-learning (Latent-RAG) validation -------------------------------
echo "--- [4/6] self-learning validation (WRITE/TEST/CONTROL) ---"
python -m alps.evaluation.self_learning_validation \
    --model-path "$MODEL" --data-path "$DATA" --frame-skip "$FRAME_SKIP"

# ---- 5. FULL hierarchy with K-frame TEMPORAL history (LeWM-style) ------------
HIER_EPOCHS="${HIER_EPOCHS:-40}"
HIER_SAMPLES="${HIER_SAMPLES:-0}"          # 0 = use all windows
TEMPORAL_WINDOW="${TEMPORAL_WINDOW:-6}"    # K-frame causal history window
# LeWM-FAITHFUL: encoder trained ONLY by feature prediction + collapse prevention,
# NO position/dynamics labels (the latent is read out by a FROZEN probe at eval).
# Default ON; set SELF_SUP=0 for the position-grounded (weakly-supervised) variant.
SELF_SUP="${SELF_SUP:-1}"
SELFSUP_FLAG=""
if [ "$SELF_SUP" = "1" ]; then SELFSUP_FLAG="--self-supervised"; fi
TEMPORAL_MODEL="results/two_rooms/validation/temporal_world_model.pt"
if [ -f "$TEMPORAL_MODEL" ] && [ -z "${RETRAIN:-}" ]; then
  echo "--- [5/6] temporal model already exists at $TEMPORAL_MODEL (skipping; set RETRAIN=1 to force) ---"
else
  echo "--- [5/6] training TEMPORAL hierarchy (self_sup=$SELF_SUP: strategic VQ + tactical MoE + goal head + RAG) ---"
  python -m alps.training.train_temporal \
      --data-path "$DATA" --epochs "$HIER_EPOCHS" --d-model "$DMODEL" \
      --enc-depth "$ENC_DEPTH" --enc-heads "$ENC_HEADS" --window "$TEMPORAL_WINDOW" \
      --num-codes "$NUM_CODES" --num-experts "$NUM_EXPERTS" --active-experts "$ACTIVE_EXPERTS" \
      --limit-samples "$HIER_SAMPLES" --save-model $SELFSUP_FLAG \
      --out "$TEMPORAL_MODEL"
fi

# ---- 5b. G1 LINEAR IDENTIFIABILITY: self-supervised vs supervised, head-to-head ----
# Trains TWO encoders (SSL: no labels / SUP: pos+dyn labels) on the same config and
# reports frozen-probe decode error for each. The gap quantifies how much the pure
# unsupervised representation gives up (LeWM: little -> position is linearly
# identifiable from the SSL latent). Doubles temporal training; set G1_COMPARE=0 to skip.
G1_COMPARE="${G1_COMPARE:-1}"
if [ "$G1_COMPARE" = "1" ]; then
  echo "--- [5b] G1 linear identifiability (SSL vs SUP) ---"
  python -m alps.evaluation.g1_identifiability \
      --data-path "$DATA" --epochs "$HIER_EPOCHS" --d-model "$DMODEL" \
      --enc-depth "$ENC_DEPTH" --enc-heads "$ENC_HEADS" --window "$TEMPORAL_WINDOW" \
      --num-codes "$NUM_CODES" --eval-samples 6000
fi

# ---- 6. FOUR-BRAIN gates, SIMPLE mode (operative/+strategic/+tactical ablation)
echo "--- [6/7] FOUR-BRAIN validation, simple mode (G1/G_str/G_tac/G_roll/G_4brain) ---"
python -m alps.evaluation.validate_temporal \
    --model-path "$TEMPORAL_MODEL" \
    --data-path "$DATA" --n-episodes "$N_EPISODES_EVAL" --coarse-k "$COARSE_K"

# ---- 7. FOUR-BRAIN gates, COMPLEX mode (key->door->goal: the decisive test) ---
# Greedy operative CANNOT represent "fetch key, then goal"; the strategic graph +
# tactical rough-region refinement is what threads it. Trained on BFS-optimal +
# random complex data (hazards off -> isolates the System-2 routing problem).
CX_EPISODES="${CX_EPISODES:-2500}"
CX_BFS_FRACTION="${CX_BFS_FRACTION:-0.5}"
CX_DATA="data/two_rooms/trajectories_complex.pt"
CX_MODEL="results/two_rooms/validation/temporal_world_model_complex.pt"
if [ ! -f "$CX_DATA" ]; then
  echo "--- [7/7] generating $CX_EPISODES-episode COMPLEX dataset (BFS-optimal + random) ---"
  python -m alps.benchmarks.two_rooms.generate_complex \
      --save-path "$CX_DATA" --num-episodes "$CX_EPISODES" --bfs-fraction "$CX_BFS_FRACTION"
fi
if [ -f "$CX_MODEL" ] && [ -z "${RETRAIN:-}" ]; then
  echo "--- [7/7] complex temporal model exists at $CX_MODEL (skipping; set RETRAIN=1 to force) ---"
else
  echo "--- [7/7] training COMPLEX temporal hierarchy (self_sup=$SELF_SUP) ---"
  python -m alps.training.train_temporal \
      --data-path "$CX_DATA" --epochs "$HIER_EPOCHS" --d-model "$DMODEL" \
      --enc-depth "$ENC_DEPTH" --enc-heads "$ENC_HEADS" --window "$TEMPORAL_WINDOW" \
      --num-codes "$NUM_CODES" --num-experts "$NUM_EXPERTS" --active-experts "$ACTIVE_EXPERTS" \
      --limit-samples "$HIER_SAMPLES" --save-model $SELFSUP_FLAG --out "$CX_MODEL"
fi
echo "--- [7/7] FOUR-BRAIN validation, COMPLEX mode (key-gated) ---"
python -m alps.evaluation.validate_temporal --complex \
    --model-path "$CX_MODEL" --data-path "$CX_DATA" \
    --n-episodes "$N_EPISODES_EVAL" --coarse-k "$COARSE_K"

# ---- 8. FOURTH BRAIN (H8/H9/H10) + MoE specialization (H11) -------------------
# Self-monitoring -> escalation -> fallback (RSRA loop, minimal form) and the
# tactical-expert causal knockout. Set FOURTH_BRAIN=0 to skip.
FOURTH_BRAIN="${FOURTH_BRAIN:-1}"
FB_CAL="${FB_CAL:-60}"; FB_EVAL="${FB_EVAL:-100}"
if [ "$FOURTH_BRAIN" = "1" ]; then
  echo "--- [8a] Fourth Brain, simple mode (monitors/escalation/fallback) ---"
  python -m alps.evaluation.fourth_brain \
      --model-path "$TEMPORAL_MODEL" --data-path "$DATA" \
      --n-cal "$FB_CAL" --n-eval "$FB_EVAL"
  echo "--- [8b] Fourth Brain, COMPLEX mode ---"
  python -m alps.evaluation.fourth_brain --complex \
      --model-path "$CX_MODEL" --data-path "$CX_DATA" \
      --n-cal "$FB_CAL" --n-eval "$FB_EVAL"
  echo "--- [8c] MoE expert specialization, simple + complex ---"
  python -m alps.evaluation.moe_specialization \
      --model-path "$TEMPORAL_MODEL" --data-path "$DATA"
  python -m alps.evaluation.moe_specialization --complex \
      --model-path "$CX_MODEL" --data-path "$CX_DATA"
  echo "--- [8d] abstraction-layer gates: strategic/tactical predict latent + emit goals ---"
  python -m alps.evaluation.validate_abstraction \
      --model-path "$TEMPORAL_MODEL" --data-path "$DATA"
fi

echo "================ DONE ================"
echo "Artifacts:"
echo "  results/two_rooms/validation/repr_decoder_gate_trained_fs${FRAME_SKIP}.json   (G1/G2)"
echo "  results/two_rooms/ablation/ladder_metrics.json + ladder_success.png + latent_graph.png"
echo "  results/two_rooms/validation/self_learning_validation.json"
echo "  results/two_rooms/validation/g1_identifiability.json       (SSL vs SUP frozen-probe G1 + collapse)"
echo "  results/two_rooms/validation/temporal_gates.json           (simple Four-Brain: G1/G_str/G_tac/G_roll/G_4brain)"
echo "  results/two_rooms/validation/temporal_gates_complex.json    (COMPLEX Four-Brain: key->door->goal 3-tier ablation)"
echo "  results/two_rooms/validation/fourth_brain{,_complex}.json   (H8 monitoring / H9 escalation / H10 fallback)"
echo "  results/two_rooms/validation/moe_specialization{,_complex}.json (H11 expert routing + knockout matrix)"
echo "  results/two_rooms/validation/abstraction_gates.json        (strategic/tactical latent prediction + goal emission)"
