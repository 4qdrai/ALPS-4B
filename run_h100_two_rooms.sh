#!/bin/bash
# ============================================================
# ALPS-4B Two Rooms Benchmark — H100 RunPod Full Pipeline
# ============================================================
# This script runs the complete pipeline:
#   1. Setup environment
#   2. Generate training data (5000 episodes)
#   3. Train ALPS-4B on Two Rooms (50 epochs)
#   4. Run inference with visualization
#   5. Push results back to GitHub
#
# Usage on RunPod:
#   1. Launch an H100 pod with PyTorch 2.x image
#   2. Open terminal
#   3. Run: bash run_h100_two_rooms.sh
# ============================================================

set -e  # Exit on any error

# Ensure python knows where the src folder is
export PYTHONPATH=src

echo "============================================================"
echo "  ALPS-4B Two Rooms Benchmark — H100 Pipeline"
echo "============================================================"

# --- 1. SETUP ---
echo ""
echo "[1/5] Setting up environment..."

# Clone the repo
cd /workspace
if [ ! -d "4B-JEPA" ]; then
    git clone https://github.com/4qdrai/4B-JEPA.git
    cd 4B-JEPA
else
    cd 4B-JEPA
    git pull
fi

# Set absolute PYTHONPATH so Python always finds the 'alps' package
export PYTHONPATH=$(pwd)/src

# Install dependencies
pip install torch torchvision --quiet 2>/dev/null || true  # Usually pre-installed on RunPod
pip install matplotlib scikit-learn pillow --quiet

# Verify CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo "[1/5] Environment ready!"

# --- 2. GENERATE DATA ---
echo ""
if [ -f "data/two_rooms/trajectories.pt" ]; then
    echo "[2/5] Found existing training data at data/two_rooms/trajectories.pt. Skipping generation."
else
    echo "[2/5] Generating training data (5000 episodes × 100 steps)..."
    python -m alps.benchmarks.two_rooms.data_generator \
        --num-episodes 5000 \
        --max-steps 100 \
        --save-path data/two_rooms/trajectories.pt
    echo "[2/5] Data generation complete!"
fi

# --- 3. TRAIN ---
echo ""
echo "[3/5] Training ALPS-4B on Two Rooms (10 epochs, d_model=384, H100 GPU)..."
python -m alps.benchmarks.two_rooms.train_two_rooms \
    --epochs 10 \
    --d-model 384 \
    --batch-size 32 \
    --lr 1e-3 \
    --lambda-sigreg 0.6 \
    --data-path data/two_rooms/trajectories.pt \
    --save-dir results/two_rooms \
    --device cuda
echo "[3/5] Training complete!"

# --- 4. EVALUATE & VISUALIZE ---
echo ""
echo "[4/5] Running evaluation and generating visualizations..."
python -m alps.benchmarks.two_rooms.evaluate_two_rooms \
    --model-path results/two_rooms/two_rooms_model.pt \
    --data-path data/two_rooms/trajectories.pt \
    --save-dir results/two_rooms/figures \
    --d-model 384
echo "[4/5] Evaluation complete!"

# --- 5. PUSH RESULTS ---
echo ""
echo "[5/5] Pushing results to GitHub..."

# Configure git
git config user.email "alps4b@runpod.io"
git config user.name "ALPS-4B H100 Runner"

# Add results (figures, metrics, model checkpoint)
git add results/two_rooms/ -f
git add data/two_rooms/trajectories.pt -f 2>/dev/null || true  # May be too large

git commit -m "feat(two-rooms): Add Two Rooms benchmark training results

Trained on H100 GPU for 50 epochs:
- Training metrics: results/two_rooms/training_log.json
- Model checkpoint: results/two_rooms/two_rooms_model.pt
- Visualizations: results/two_rooms/figures/
  - trajectory_overlay.png: System 1/2 activation paths
  - energy_landscape.png: Prediction difficulty heatmap
  - latent_clustering.png: t-SNE room separation
  - vq_codebook_usage.png: Strategic concept mapping
  - prediction_comparison.png: Predicted vs actual positions
  - planning_metrics.json: Same-room vs cross-room success rates"

git push

echo ""
echo "============================================================"
echo "  PIPELINE COMPLETE!"
echo ""
echo "  Results pushed to: https://github.com/4qdrai/4B-JEPA"
echo "  Figures saved in:  results/two_rooms/figures/"
echo "============================================================"

# List generated files
echo ""
echo "Generated files:"
find results/two_rooms/ -type f | sort
