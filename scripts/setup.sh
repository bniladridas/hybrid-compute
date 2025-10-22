#!/bin/bash

# Hybrid Compute Setup Script
# Cross-platform dependency installation with colorized logs.

set -e

# --- Colors ---
GREEN="\033[1;32m"
BLUE="\033[1;34m"
YELLOW="\033[1;33m"
RED="\033[1;31m"
RESET="\033[0m"

echo -e "${BLUE}Starting setup for Hybrid Compute...${RESET}"
echo -e "${YELLOW}Installing system dependencies...${RESET}"

# --- macOS setup ---
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${BLUE}Detected macOS.${RESET}"

    # Check Homebrew
    if ! command -v brew >/dev/null 2>&1; then
        echo -e "${RED}Homebrew not found. Please install Homebrew first.${RESET}"
        exit 1
    fi

    brew uninstall opencv 2>/dev/null || true
    brew install --cask miniconda || true

    # Dynamic conda detection
    CONDA_PATH=$(find /usr/local /opt/homebrew -type f -name conda 2>/dev/null | head -n1)
    if [[ -f "$CONDA_PATH" ]]; then
        eval "$($CONDA_PATH shell.bash hook)"
        conda init bash
        conda install -c conda-forge opencv cmake imagemagick -y
    else
        echo -e "${RED}Conda not found on this system.${RESET}"
        exit 1
    fi

# --- Linux setup ---
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo -e "${BLUE}Detected Linux (Ubuntu).${RESET}"
    sudo apt-get update -y
    sudo apt-get install -y cmake libopencv-dev build-essential imagemagick wget

    echo -e "${YELLOW}Installing CUDA toolkit (optional for GPU acceleration)...${RESET}"
    TMP_CUDA="/tmp/cuda-keyring.deb"
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O "$TMP_CUDA"
    sudo dpkg -i "$TMP_CUDA"
    sudo apt-get update -y
    sudo apt-get install -y cuda-toolkit-11-8 || echo -e "${RED}CUDA installation skipped or failed.${RESET}"
    rm -f "$TMP_CUDA"

# --- Windows setup ---
elif [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "cygwin"* ]]; then
    echo -e "${BLUE}Detected Windows environment.${RESET}"
    if ! command -v choco >/dev/null 2>&1; then
        echo -e "${RED}Chocolatey not found. Please install Chocolatey first.${RESET}"
        exit 1
    fi
    choco install cmake opencv imagemagick -y || echo -e "${RED}Some packages may not have installed.${RESET}"

# --- Unsupported OS ---
else
    echo -e "${RED}Unsupported OS: $OSTYPE${RESET}"
    exit 1
fi

# --- Python setup ---
echo -e "${YELLOW}Installing Python dependencies...${RESET}"

# Check Python >= 3.9
if command -v python3 >/dev/null 2>&1; then
    PY_OK=$(python3 -c "import sys; print(int((sys.version_info.major, sys.version_info.minor) >= (3,9)))")
    if [ "$PY_OK" -eq 0 ]; then
        PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        echo -e "${RED}Python 3.9 or higher is required. Found: $PY_VER${RESET}"
        exit 1
    fi
else
    echo -e "${RED}Python3 not found. Please install Python 3.9+${RESET}"
    exit 1
fi

# Install pip dependencies
if command -v pip3 >/dev/null 2>&1; then
    pip3 install --upgrade pip
    pip3 install --upgrade -r requirements.txt --user --no-warn-script-location
else
    echo -e "${RED}pip not found. Please install pip.${RESET}"
    exit 1
fi

echo -e "${GREEN}Setup complete.${RESET}"
