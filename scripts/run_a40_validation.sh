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
DMODEL="${DMODEL:-192}"           # latent width (must be divisible by 4)
EPOCHS="${EPOCHS:-100}"
POS_WEIGHT="${POS_WEIGHT:-1.0}"
DYN_WEIGHT="${DYN_WEIGHT:-1.0}"
SIGREG_SLICES="${SIGREG_SLICES:-512}"
PROBE_EPOCHS="${PROBE_EPOCHS:-150}"
N_EPISODES_EVAL="${N_EPISODES_EVAL:-200}"
GRAPH_K="${GRAPH_K:-24}"
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
echo "--- [2/4] training ReprWorldModel + gates G1/G2 ---"
python -m alps.evaluation.repr_decoder_gate train \
    --data-path "$DATA" --frame-skip "$FRAME_SKIP" --d-model "$DMODEL" \
    --epochs "$EPOCHS" --pos-weight "$POS_WEIGHT" --dyn-weight "$DYN_WEIGHT" \
    --sigreg-slices "$SIGREG_SLICES" --probe-epochs "$PROBE_EPOCHS" --save-model

# ---- 3. Planning ablation ladder (the edge evidence) ------------------------
echo "--- [3/4] ablation ladder (rungs 0/2/4a/4b/5) ---"
python -m alps.benchmarks.two_rooms.run_ablation_ladder \
    --model-path "$MODEL" --data-path "$DATA" --frame-skip "$FRAME_SKIP" \
    --n-episodes "$N_EPISODES_EVAL" --graph-k "$GRAPH_K"

# ---- 4. Self-learning (Latent-RAG) validation -------------------------------
echo "--- [4/4] self-learning validation (WRITE/TEST/CONTROL) ---"
python -m alps.evaluation.self_learning_validation \
    --model-path "$MODEL" --data-path "$DATA" --frame-skip "$FRAME_SKIP"

echo "================ DONE ================"
echo "Artifacts:"
echo "  results/two_rooms/validation/repr_decoder_gate_trained_fs${FRAME_SKIP}.json   (G1/G2)"
echo "  results/two_rooms/ablation/ladder_metrics.json + ladder_success.png + latent_graph.png"
echo "  results/two_rooms/validation/self_learning_validation.json"
