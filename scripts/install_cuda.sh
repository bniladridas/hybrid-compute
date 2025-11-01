#!/bin/bash
set -euo pipefail

# Default values
CUDA_VERSION=${1:-12.6}
CUDA_ARCHITECTURES=${2:-75,80,86}
INSTALL_DEPS=${3:-true}

# Convert comma-separated architectures to semicolon-separated for CMake
CMAKE_ARCHITECTURES=$(echo "$CUDA_ARCHITECTURES" | tr ',' ';')

# Get Ubuntu version
UBUNTU_VERSION=$(lsb_release -rs)
UBUNTU_CODENAME=$(lsb_release -cs)

# Log function
log() {
    echo -e "\n[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Error handling
handle_error() {
    local exit_code=$?
    log "❌ Error in line $1"
    exit $exit_code
}

trap 'handle_error $LINENO' ERR

# Check if running as root
if [ "$(id -u)" -eq 0 ]; then
    log "This script should not be run as root. Please run as a regular user with sudo privileges."
    exit 1
fi

log "Starting CUDA ${CUDA_VERSION} installation on Ubuntu ${UBUNTU_VERSION} (${UBUNTU_CODENAME})"

# Install dependencies
if [ "$INSTALL_DEPS" = "true" ]; then
    log "Installing required dependencies..."
    sudo apt-get update
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        software-properties-common \
        wget \
        gnupg \
        ca-certificates \
        curl \
        git \
        lsb-release
fi

# Determine the appropriate repository for the Ubuntu version
case "$UBUNTU_CODENAME" in
    jammy)
        REPO_DISTRO="ubuntu2204"
        ;;
    focal)
        REPO_DISTRO="ubuntu2004"
        ;;
    *)
        log "Ubuntu version ${UBUNTU_VERSION} (${UBUNTU_CODENAME}) is not officially supported. Defaulting to Ubuntu 22.04 repository."
        REPO_DISTRO="ubuntu2204"
        ;;
esac

# Install CUDA repository keyring
log "Setting up CUDA repository..."
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
KEYRING_PKG="cuda-keyring_1.1-1_all.deb"
KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/${REPO_DISTRO}/x86_64/${KEYRING_PKG}"

curl -fsSL "$KEYRING_URL" -o /tmp/cuda-keyring.deb
sudo dpkg -i /tmp/cuda-keyring.deb
rm -f /tmp/cuda-keyring.deb

# Install CUDA
log "Installing CUDA ${CUDA_VERSION}..."
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    "cuda-compiler-${CUDA_MAJOR}-${CUDA_MINOR}" \
    "cuda-libraries-${CUDA_MAJOR}-${CUDA_MINOR}" \
    "cuda-nvtx-${CUDA_MAJOR}-${CUDA_MINOR}" \
    "cuda-nvml-dev-${CUDA_MAJOR}-${CUDA_MINOR}" \
    "cuda-command-line-tools-${CUDA_MAJOR}-${CUDA_MINOR}" \
    || { log "Failed to install CUDA packages"; exit 1; }

# Set up environment variables
echo "# CUDA" >> ~/.bashrc
echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
echo "export PATH=\"\${CUDA_HOME}/bin:\${PATH}\"" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\"\${CUDA_HOME}/lib64:\${LD_LIBRARY_PATH:-}\"" >> ~/.bashrc

# Export for current shell
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# For GitHub Actions
if [ -n "${GITHUB_ENV:-}" ]; then
    echo "CUDA_HOME=${CUDA_HOME}" >> "$GITHUB_ENV"
    echo "PATH=${CUDA_HOME}/bin:${PATH}" >> "$GITHUB_ENV"
    echo "LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}" >> "$GITHUB_ENV"
    echo "CUDACXX=${CUDA_HOME}/bin/nvcc" >> "$GITHUB_ENV"
    echo "CMAKE_CUDA_ARCHITECTURES=${CMAKE_ARCHITECTURES}" >> "$GITHUB_ENV"
fi

# Verify installation
log "Verifying CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    log "❌ nvcc not found. CUDA installation may have failed."
    exit 1
fi

NVCC_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
log "✅ CUDA ${NVCC_VERSION} installed successfully"

# Print CUDA information
log "CUDA Installation Summary:"
echo "  - CUDA Version: ${NVCC_VERSION}"
echo "  - CUDA Home: ${CUDA_HOME}"
echo "  - nvcc Path: $(which nvcc)"
echo "  - Target Architectures: ${CUDA_ARCHITECTURES}"

# Clean up
log "Cleaning up..."
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*

log "CUDA installation completed successfully!"
log "To use CUDA in your current shell, run: source ~/.bashrc"
