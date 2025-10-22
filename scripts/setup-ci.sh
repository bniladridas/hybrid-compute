#!/bin/bash

# CI Setup Script for Dependencies
# This script installs all dependencies required for CI workflows
# Usage: ./scripts/setup-ci.sh [platform]
# Platforms: linux, macos, windows

set -e

PLATFORM=${1:-linux}

echo "Setting up dependencies for $PLATFORM..."

if [ "$PLATFORM" == "linux" ]; then
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends cmake libopencv-dev build-essential imagemagick
    # CUDA for linux CI
    if [ "$CUDA" == "true" ]; then
        wget -q \
          https://developer.download.nvidia.com/compute/cuda/repos/ \
          ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
          -O /tmp/cuda-keyring.deb
        sudo dpkg -i /tmp/cuda-keyring.deb
        sudo apt-get update
        sudo apt-get install -y --no-install-recommends cuda-toolkit-11-8 || \
          echo "CUDA installation skipped"
    fi
elif [ "$PLATFORM" == "macos" ]; then
    brew install imagemagick
    # Conda setup assumed handled elsewhere
elif [ "$PLATFORM" == "windows" ]; then
    # Choco installs assumed
    echo "Windows dependencies via choco"
fi

# Python dependencies (common)
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install black isort ruff yamllint pre-commit pytest coverage pytest-cov mypy

echo "Setup complete for $PLATFORM"
