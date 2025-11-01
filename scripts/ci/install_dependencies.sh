#!/bin/bash
set -euo pipefail

# Function to check if running with sudo
check_sudo() {
    if [ "$(id -u)" -ne 0 ]; then
        echo "This script requires root privileges. Please run with sudo or as root."
        exit 1
    fi
}

# Install system dependencies for Linux
install_linux_deps() {
    echo "Installing Linux dependencies..."
    # Check if running with sudo in CI environment
    if [ -z "${CI:-}" ] && [ "$(id -u)" -ne 0 ]; then
        echo "This script requires root privileges. Please run with sudo or as root."
        exit 1
    fi

    # Use sudo if not root
    local SUDO=""
    if [ "$(id -u)" -ne 0 ]; then
        SUDO=sudo
    fi

    $SUDO apt-get update && $SUDO apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
        libopencv-dev \
        git \
        pkg-config \
        wget \
        python3 \
        python3-pip \
        python3-dev \
        clang \
        llvm \
        lld \
        && $SUDO rm -rf /var/lib/apt/lists/*
}

# Install Python dependencies
install_python_deps() {
    echo "Installing Python dependencies..."

    # Use --user flag to avoid permission issues with system pip
    python3 -m pip install --user --upgrade pip setuptools wheel

    # Add user's Python packages to PATH if not already there
    if [[ ":$PATH:" != *"$HOME/.local/bin"* ]]; then
        export PATH="$HOME/.local/bin:$PATH"
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    fi

    # Install project dependencies
    pip install --user -r requirements.txt

    # Verify installation
    if ! python3 -c "import pkg_resources; pkg_resources.require(open('requirements.txt',mode='r'))" &>/dev/null; then
        echo "Failed to verify Python dependencies. Please check the logs."
        exit 1
    fi
}

# Install CUDA (if needed)
install_cuda() {
    if [ "${USE_CUDA:-0}" = "1" ]; then
        echo "Installing CUDA dependencies..."

        # Use sudo if not root
        local SUDO=""
        if [ "$(id -u)" -ne 0 ]; then
            SUDO=sudo
        fi

        # Install CUDA repository key
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
        $SUDO dpkg -i cuda-keyring_1.1-1_all.deb
        $SUDO apt-get update

        # Install CUDA toolkit
        $SUDO apt-get install -y --no-install-recommends cuda-toolkit-12-3

        # Add CUDA to PATH
        echo 'export PATH="/usr/local/cuda/bin:$PATH"' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
        export PATH="/usr/local/cuda/bin:$PATH"
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

        # Verify installation
        nvcc --version || {
            echo "Failed to verify CUDA installation. Please check the logs."
            exit 1
        }
    fi
}

# Install CodeQL
install_codeql() {
    echo "Installing CodeQL..."
    # Install GitHub CLI if not already installed
    if ! command -v gh &> /dev/null; then
        type -p curl >/dev/null || (sudo apt update && sudo apt install curl -y)
        curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
        && sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
        && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
        && sudo apt update \
        && sudo apt install gh -y
    fi

    # Install CodeQL
    gh extension install github/gh-codeql --force
    gh codeql install

    # Add CodeQL to PATH
    echo 'export PATH="$HOME/.local/share/github/codeql:$PATH"' >> ~/.bashrc
    export PATH="$HOME/.local/share/github/codeql:$PATH"

    # Verify installation
    codeql version
}

# Main function
main() {
    install_linux_deps
    install_python_deps
    install_cuda
    install_codeql
}

# Run main function
main "$@"
