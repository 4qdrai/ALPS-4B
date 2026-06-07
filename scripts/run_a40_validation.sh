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
echo "--- [2/4] training ReprWorldModel + gates G1/G2 ---"
python -m alps.evaluation.repr_decoder_gate train \
    --data-path "$DATA" --frame-skip "$FRAME_SKIP" --d-model "$DMODEL" \
    --enc-depth "$ENC_DEPTH" --enc-heads "$ENC_HEADS" \
    --epochs "$EPOCHS" --pos-weight "$POS_WEIGHT" --dyn-weight "$DYN_WEIGHT" \
    --sigreg-slices "$SIGREG_SLICES" --probe-epochs "$PROBE_EPOCHS" --save-model

# ---- 3. Planning ablation ladder (the edge evidence) ------------------------
echo "--- [3/4] ablation ladder (rungs 0/2/4a/4b/5) ---"
python -m alps.benchmarks.two_rooms.run_ablation_ladder \
    --model-path "$MODEL" --data-path "$DATA" --frame-skip "$FRAME_SKIP" \
    --n-episodes "$N_EPISODES_EVAL" --graph-k "$GRAPH_K"

# ---- 4. Self-learning (Latent-RAG) validation -------------------------------
echo "--- [4/6] self-learning validation (WRITE/TEST/CONTROL) ---"
python -m alps.evaluation.self_learning_validation \
    --model-path "$MODEL" --data-path "$DATA" --frame-skip "$FRAME_SKIP"

# ---- 5. FULL hierarchy: train learned strategic/tactical layers --------------
HIER_EPOCHS="${HIER_EPOCHS:-60}"
HIER_SAMPLES="${HIER_SAMPLES:-0}"          # 0 = use all multi-scale samples
HIER_SAMPLE_STRIDE="${HIER_SAMPLE_STRIDE:-1}"  # 1 = densest sampling (most operative transitions)
echo "--- [5/6] training FULL hierarchy (strategic VQ + tactical MoE + goal head + RAG) ---"
python -m alps.training.train_hier \
    --data-path "$DATA" --epochs "$HIER_EPOCHS" --d-model "$DMODEL" \
    --enc-depth "$ENC_DEPTH" --enc-heads "$ENC_HEADS" \
    --num-codes "$NUM_CODES" --num-experts "$NUM_EXPERTS" --active-experts "$ACTIVE_EXPERTS" \
    --sample-stride "$HIER_SAMPLE_STRIDE" --limit-samples "$HIER_SAMPLES" --save-model \
    --out results/two_rooms/validation/hier_world_model.pt

# ---- 6. Per-layer hierarchy gates (G_op/G_str/G_tac/G_goals/G_rag) -----------
echo "--- [6/6] per-layer hierarchy validation ---"
python -m alps.evaluation.validate_hierarchy \
    --model-path results/two_rooms/validation/hier_world_model.pt \
    --data-path "$DATA" --n-episodes "$N_EPISODES_EVAL"

echo "================ DONE ================"
echo "Artifacts:"
echo "  results/two_rooms/validation/repr_decoder_gate_trained_fs${FRAME_SKIP}.json   (G1/G2)"
echo "  results/two_rooms/ablation/ladder_metrics.json + ladder_success.png + latent_graph.png"
echo "  results/two_rooms/validation/self_learning_validation.json"
echo "  results/two_rooms/validation/hierarchy_gates.json   (G_op/G_str/G_tac/G_goals/G_rag)"
