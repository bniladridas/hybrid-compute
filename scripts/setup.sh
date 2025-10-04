#!/bin/bash

# Setup script for Hybrid Compute

set -e

echo "Installing dependencies..."

if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    brew uninstall opencv || true
    brew install --cask miniconda
    eval "$(~/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    conda install -c conda-forge opencv cmake imagemagick -y
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Ubuntu
    sudo apt-get update
    sudo apt-get install -y cmake libopencv-dev build-essential imagemagick
else
    # Windows (assuming MSYS or similar)
    choco install cmake opencv imagemagick
fi

echo "Installing Python dependencies..."
pip install -r requirements.txt --user
python3 -m pip install opencv-python --user

echo "Setup complete."