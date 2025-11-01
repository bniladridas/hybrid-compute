#!/bin/bash
set -euo pipefail

# Install system dependencies for Linux
install_linux_deps() {
    echo "Installing Linux dependencies..."
    apt-get update && apt-get install -y --no-install-recommends \
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
        && rm -rf /var/lib/apt/lists/*
}

# Install Python dependencies
install_python_deps() {
    echo "Installing Python dependencies..."
    python3 -m pip install --upgrade pip
    pip install -r requirements.txt
}

# Install CUDA (if needed)
install_cuda() {
    if [ "${USE_CUDA:-0}" = "1" ]; then
        echo "Installing CUDA dependencies..."
        # Install CUDA repository key
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
        dpkg -i cuda-keyring_1.1-1_all.deb
        apt-get update

        # Install CUDA toolkit
        apt-get install -y --no-install-recommends cuda-toolkit-12-3

        # Add CUDA to PATH
        echo 'export PATH="/usr/local/cuda/bin:$PATH"' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
        export PATH="/usr/local/cuda/bin:$PATH"
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

        # Verify installation
        nvcc --version
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
