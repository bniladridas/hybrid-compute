#!/bin/bash
set -e

export PATH=/usr/bin:/bin:$PATH

echo "Installing CUDA..."

URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
/usr/bin/curl -s -o /tmp/cuda-keyring.deb "$URL"
/usr/bin/sudo /usr/bin/apt install -y /tmp/cuda-keyring.deb
echo "deb http://archive.ubuntu.com/ubuntu jammy main" | /usr/bin/sudo /usr/bin/tee /etc/apt/sources.list.d/jammy.list
/usr/bin/sudo /usr/bin/apt-get update
/usr/bin/sudo /usr/bin/apt-get install -y --no-install-recommends cuda-compiler-12-6 cuda-libraries-12-6 || echo "CUDA installation skipped"
export PATH=/usr/local/cuda/bin:$PATH
echo "CUDACXX=nvcc" >> $GITHUB_ENV
echo "CMAKE_CUDA_ARCHITECTURES=75;80;86" >> $GITHUB_ENV

echo "Verifying CUDA installation..."
which nvcc
nvcc --version
