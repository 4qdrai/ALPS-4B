#!/bin/bash
# runpod_setup.sh - Automated H100 CUDA environment setup for ALPS-4B

echo "=== ALPS-4B: RunPod H100 Environment Setup ==="

# 1. Update system packages
echo "Updating system package repositories..."
sudo apt-get update -y && sudo apt-get upgrade -y

# 2. Install basic system dependencies
echo "Installing system utilities..."
sudo apt-get install -y git build-essential python3-dev python3-pip python3-venv

# 3. Check for CUDA availability
echo "Checking CUDA version..."
nvcc --version || echo "Warning: nvcc not found on PATH. Make sure CUDA drivers are active."

# 4. Set up Python virtual environment
echo "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 5. Upgrade pip and packaging tools
pip install --upgrade pip setuptools wheel

# 6. Install PyTorch with native CUDA 12.1+ support
echo "Installing CUDA-accelerated PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 7. Install ALPS package in editable development mode with dev dependencies
echo "Installing ALPS package and test dependencies..."
pip install -e ".[dev]"

echo "=== Setup Complete! ==="
echo "To activate the environment: source venv/bin/activate"
echo "To run test suite: pytest -v"
echo "To start training: bash scripts/run_training.sh"
