#!/bin/bash
# run_training.sh - Launcher for ALPS-4B training pipeline

echo "=== ALPS-4B: Launching Unsupervised Video JEPA Training ==="

# Set PYTHONPATH to include source root
export PYTHONPATH="src"

# Activate virtual environment if present
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run the PyTorch training pipeline
python3 src/alps/training/train.py
