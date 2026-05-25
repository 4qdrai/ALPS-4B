#!/bin/bash
# download_ucf101.sh - Academic dataset retrieval for ALPS-4B training
# Downloads the UCF101 Action Recognition Dataset (~6.5 GB) and extracts it.

set -e

echo "=== ALPS-4B: UCF101 Dataset Downloader ==="
echo "This script retrieves the UCF101 Action Recognition Dataset required for training."

# 1. Create data directories
DATA_DIR="data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# 2. Define URLs
UCF101_URL="https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
ANNO_URL="https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"

# 3. Check for existing downloads
if [ -f "UCF101.rar" ]; then
    echo "UCF101.rar already exists. Skipping download."
else
    echo "Downloading UCF101 dataset (~6.5 GB)..."
    echo "Note: This can take some time depending on your connection."
    curl -k -L -O "$UCF101_URL" || wget --no-check-certificate "$UCF101_URL"
fi

if [ -f "UCF101TrainTestSplits-RecognitionTask.zip" ]; then
    echo "Train/Test splits already exist. Skipping download."
else
    echo "Downloading Train/Test splits..."
    curl -k -L -O "$ANNO_URL" || wget --no-check-certificate "$ANNO_URL"
fi

# 4. Extraction
echo "Extracting Train/Test splits..."
unzip -q -o UCF101TrainTestSplits-RecognitionTask.zip

echo "Extracting UCF101 videos..."
echo "Note: This requires 'unrar' or 'rar' installed on your system."
if command -v unrar &> /dev/null; then
    unrar x -q UCF101.rar
elif command -v 7z &> /dev/null; then
    7z x UCF101.rar
else
    echo "Warning: Neither unrar nor 7z found. Please extract UCF101.rar manually."
fi

echo "=== Dataset Preparation Complete! ==="
echo "UCF101 dataset extracted to data/UCF101/"
