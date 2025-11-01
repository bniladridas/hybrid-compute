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
    if [ "${USE_CUDA:-0}" != "1" ]; then
        echo "CUDA installation skipped (USE_CUDA not set to 1)"
        return 0
    fi

    echo "Installing CUDA dependencies..."

    # Use sudo if not root
    local SUDO=""
    if [ "$(id -u)" -ne 0 ]; then
        SUDO=sudo
    fi

    # Install basic CUDA dependencies
    $SUDO apt-get update
    $SUDO apt-get install -y --no-install-recommends \
        build-essential \
        software-properties-common \
        wget \
        gnupg2 \
        lsb-release

    # Install CUDA repository key
    if [ ! -f "/etc/apt/sources.list.d/cuda.list" ]; then
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
        $SUDO dpkg -i cuda-keyring_1.1-1_all.deb
        $SUDO apt-get update
    fi

    # Install essential CUDA packages (without cuDNN)
    echo "Installing CUDA 12.6 toolkit..."
    $SUDO apt-get install -y --no-install-recommends \
        cuda-compiler-12-6 \
        cuda-libraries-dev-12-6 \
        cuda-command-line-tools-12-6 \
        cuda-cudart-dev-12-6 \
        cuda-nvcc-12-6 \
        libcublas-dev-12-6 \
        libcufft-dev-12-6 \
        libcurand-dev-12-6 \
        libcusolver-dev-12-6 \
        libcusparse-dev-12-6

    # Install cuDNN from NVIDIA's repository
    echo "Installing cuDNN..."

    # Add NVIDIA's repository key for cuDNN
    $SUDO apt-get update
    $SUDO apt-get install -y --no-install-recommends \
        libcudnn8=8.9.7.*-1+cuda12.6 \
        libcudnn8-dev=8.9.7.*-1+cuda12.6 || {
            echo "Warning: Failed to install cuDNN 8.9.7. Trying alternative installation method..."

            # Alternative method: Install from the CUDA repository
            $SUDO apt-get install -y --no-install-recommends \
                libcudnn8 \
                libcudnn8-dev \
                cuda-nvtx-12-3 \
                cuda-nvml-dev-12-3 \
                cuda-command-line-tools-12-3 \
                cuda-libraries-dev-12-3 \
                cuda-minimal-build-12-3 \
                libcublas-dev-12-3 \
                libcufft-dev-12-3 \
                libcurand-dev-12-3 \
                libcusolver-dev-12-3 \
                libcusparse-dev-12-3
        }

    # Create symlinks for backward compatibility
    $SUDO ln -sf /usr/local/cuda-12.3 /usr/local/cuda

    # Add cuDNN to library path
    echo 'export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"' >> ~/.bashrc
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

    # Add CUDA to PATH
    echo 'export PATH="/usr/local/cuda/bin:$PATH"' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

    # Verify installation
    if ! command -v nvcc &> /dev/null; then
        echo "Error: nvcc not found after CUDA installation"
        exit 1
    fi

    echo "CUDA installation complete. nvcc version:"
    nvcc --version || {
        echo "Warning: nvcc version check failed, but continuing..."
    }
}

# Install CodeQL
install_codeql() {
    echo "Installing CodeQL..."

    # Create codeql home directory
    export CODEQL_HOME="${HOME}/codeql-home"
    mkdir -p "${CODEQL_HOME}"

    # Install GitHub CLI if not already installed
    if ! command -v gh &> /dev/null; then
        echo "Installing GitHub CLI..."
        type -p curl >/dev/null || (sudo apt update && sudo apt install -y curl)
        curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | \
            sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
            && sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
            && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | \
               sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
            && sudo apt update \
            && sudo apt install -y gh
    fi

    # Install CodeQL CLI if not already installed
    if ! command -v codeql &> /dev/null; then
        echo "Downloading and installing CodeQL..."
        cd "${CODEQL_HOME}" || exit 1

        # Download and extract CodeQL
        wget -q https://github.com/github/codeql-cli-binaries/releases/latest/download/codeql-linux64.zip
        unzip -q codeql-linux64.zip
        rm codeql-linux64.zip

        # Add to PATH
        echo "export PATH=\"${CODEQL_HOME}/codeql:\$PATH\"" >> ~/.bashrc
        export PATH="${CODEQL_HOME}/codeql:${PATH}"

        # Verify the installation
        if ! "${CODEQL_HOME}/codeql/codeql" --version; then
            echo "Failed to verify CodeQL installation."
            exit 1
        fi

        cd - >/dev/null || exit 1
    fi

    # Ensure codeql is in PATH for the current shell
    if ! command -v codeql &> /dev/null; then
        export PATH="${CODEQL_HOME}/codeql:${PATH}"
    fi

    # Final verification
    if ! command -v codeql &> /dev/null; then
        echo "CodeQL installation failed. The 'codeql' command is not available."
        exit 1
    fi

    echo "CodeQL version: $(codeql --version)"
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
