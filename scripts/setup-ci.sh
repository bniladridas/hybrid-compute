#!/bin/bash

# CI Setup Script for Dependencies
# This script installs all dependencies required for CI workflows
# Usage: ./scripts/setup-ci.sh [platform]
# Platforms: linux, macos, windows

set -e

PLATFORM=${1:-linux}
CUDA=${2:-false}

echo -e "\033[32mSetting up dependencies for $PLATFORM...\033[0m"

if [ "$PLATFORM" == "linux" ]; then
    if [ "$(id -u)" -eq 0 ]; then
        SUDO=""
    else
        SUDO="sudo"
    fi
    env DEBIAN_FRONTEND=noninteractive $SUDO apt-get update
    env DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y --no-install-recommends cmake libopencv-dev build-essential imagemagick git wget python3 python3-pip
    # CUDA for linux CI
    if [ "$CUDA" == "cuda" ]; then
        if wget --tries=3 -q \
          https://developer.download.nvidia.com/compute/cuda/repos/ \
          ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
          -O /tmp/cuda-keyring.deb; then
            $SUDO dpkg -i /tmp/cuda-keyring.deb
            env DEBIAN_FRONTEND=noninteractive $SUDO apt-get update
            env DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y cuda-toolkit-12-6 || \
              echo -e "\033[33mCUDA installation skipped\033[0m"
        else
            echo -e "\033[31mCUDA keyring download failed, skipping CUDA setup\033[0m"
        fi
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
pip install black isort ruff yamllint pre-commit pytest coverage pytest-cov pytest-timeout mypy

echo -e "\033[32mSetup complete for $PLATFORM\033[0m"
