#!/bin/bash

# Hybrid Compute Runner Script

set -e

echo "Building preprocess..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Dynamic conda detection
    CONDA_PATH=$(find /usr/local /opt/homebrew -type f -name conda 2>/dev/null | head -n1)
    if [[ -f "$CONDA_PATH" ]]; then
        eval "$($CONDA_PATH shell.bash hook)"
        conda activate base
    else
        echo "Conda not found. Please install miniconda."
        exit 1
    fi
fi
mkdir -p build
cd build
cmake ..
if [[ "$OSTYPE" == "darwin"* ]]; then
    make -j$(sysctl -n hw.ncpu)
else
    make -j$(nproc)
fi
cd ..

echo "Testing cv2..."
python3 -c "import cv2; print('cv2 works:', cv2.__version__)"

echo "Running e2e test..."
mkdir -p test_images/tiles test_images/upscaled
magick convert -size 256x256 xc:red test_images/test.jpg
./build/preprocess test_images test_images/tiles
cp test_images/tiles/* test_images/upscaled/
for i in {0..15}; do mv test_images/upscaled/test_tile_$i.jpg test_images/upscaled/tile_$i.jpg; done
python3 scripts/stitch.py test_images/upscaled test_images/final_output.jpg
test -f test_images/final_output.jpg && echo "E2E test passed"

echo "Cleaning up..."
rm -rf build test_images
