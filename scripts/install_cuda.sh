#!/bin/bash
set -e

export PATH=/usr/bin:/bin:$PATH

echo "Installing CUDA..."

# Ensure curl is installed
sudo apt-get install -y curl

URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
curl -s -o /tmp/cuda-keyring.deb "$URL"
sudo apt install -y /tmp/cuda-keyring.deb
echo "deb http://archive.ubuntu.com/ubuntu jammy main" | sudo tee /etc/apt/sources.list.d/jammy.list
sudo apt-get update
sudo apt-get install -y --no-install-recommends cuda-compiler-12-6 cuda-libraries-12-6 || echo "CUDA installation skipped"
export PATH=/usr/local/cuda/bin:$PATH
echo "CUDACXX=nvcc" >> $GITHUB_ENV
echo "CMAKE_CUDA_ARCHITECTURES=75;80;86" >> $GITHUB_ENV

echo "Verifying CUDA installation..."
which nvcc
nvcc --version
