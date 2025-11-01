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
        # Add CUDA repository and install CUDA toolkit
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
        dpkg -i cuda-keyring_1.1-1_all.deb
        apt-get update
        apt-get -y install cuda-toolkit-13-0
        export PATH="/usr/local/cuda-13.0/bin:$PATH"
        export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH"
    fi
}

# Main installation function
main() {
    # Detect OS
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        case $ID in
            ubuntu|debian)
                install_linux_deps
                install_cuda
                ;;
            *)
                echo "Unsupported Linux distribution: $ID"
                exit 1
                ;;
        esac
    else
        echo "Unsupported operating system"
        exit 1
    fi

    # Install Python dependencies
    install_python_deps

    echo "Dependencies installed successfully!"
}

# Run the main function
main "$@"
