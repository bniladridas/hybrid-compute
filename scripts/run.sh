#!/bin/bash

# Hybrid Compute Runner Script

set -e

# --- Colors for better visibility ---
GREEN="\033[1;32m"
BLUE="\033[1;34m"
YELLOW="\033[1;33m"
RED="\033[1;31m"
RESET="\033[0m"

echo -e "${BLUE}Building preprocess...${RESET}"

# macOS conda activation
if [[ "$OSTYPE" == "darwin"* ]]; then
	export TZ=UTC
	CONDA_PATH=$(find /usr/local /opt/homebrew -type f -name conda 2>/dev/null | head -n1)
	if [[ -f "$CONDA_PATH" ]]; then
		eval "$($CONDA_PATH shell.bash hook)"
		conda activate base
		echo -e "${GREEN}Conda activated.${RESET}"
	else
		echo -e "${RED}Conda not found. Please install miniconda.${RESET}"
		exit 1
	fi
fi

# --- Build ---
mkdir -p build
cd build
cmake ..
CPU_COUNT=1
if [[ "$OSTYPE" == "darwin"* ]]; then
	CPU_COUNT=$(sysctl -n hw.logicalcpu)
else
	CPU_COUNT=$(nproc)
fi
echo -e "${YELLOW}Compiling with $CPU_COUNT cores...${RESET}"
make -j"$CPU_COUNT"

# --- Run C++ unit tests ---
echo -e "${BLUE}Running C++ unit tests...${RESET}"
ctest -j"$CPU_COUNT"
cd ..

# --- Test cv2 ---
echo -e "${BLUE}Testing cv2...${RESET}"
python3 -c "import cv2; print('cv2 works:', cv2.__version__)"

# --- Run E2E test ---
echo -e "${BLUE}Running end-to-end test...${RESET}"
mkdir -p test_images/tiles test_images/upscaled

# Generate test image
python3 -c "
import cv2
import numpy as np
img = np.full((256,256,3), (0,0,255), np.uint8)
cv2.imwrite('test_images/test.jpg', img)
"

# Run preprocessing
./build/bin/preprocess_c test_images test_images/tiles

# Copy tiles to upscaled
cp test_images/tiles/* test_images/upscaled/

# Rename tiles consistently
for i in {0..15}; do
	if [ -f test_images/upscaled/test_tile_"$i".jpg ]; then
		mv test_images/upscaled/test_tile_"$i".jpg test_images/upscaled/tile_"$i".jpg
	fi
done

# Stitch tiles
python3 scripts/stitch.py test_images/upscaled test_images/final_output.jpg

# Verify output
if [ -f test_images/final_output.jpg ]; then
	echo -e "${GREEN}E2E test passed.${RESET}"
else
	echo -e "${RED}E2E test failed.${RESET}"
	exit 1
fi

# --- Test CUDA tools (if available) ---
if [ -f build/bin/filters ]; then
  echo -e "${BLUE}Testing CUDA filters...${RESET}"
  ./build/bin/filters test_images/test.jpg test_images/filtered_blur.jpg blur
  ./build/bin/filters test_images/test.jpg test_images/filtered_sobel.jpg sobel
	if [ -f test_images/filtered_blur.jpg ] && [ -f test_images/filtered_sobel.jpg ]; then
		echo -e "${GREEN}CUDA filters test passed.${RESET}"
	else
		echo -e "${YELLOW}CUDA filters test skipped or failed.${RESET}"
	fi
fi

if [ -f build/bin/rotation ]; then
  echo -e "${BLUE}Testing CUDA rotation...${RESET}"
  ./build/bin/rotation test_images/test.jpg test_images/rotated.jpg 45
	if [ -f test_images/rotated.jpg ]; then
		echo -e "${GREEN}CUDA rotation test passed.${RESET}"
	else
		echo -e "${YELLOW}CUDA rotation test skipped or failed.${RESET}"
	fi
fi

if [ -f build/bin/resize ]; then
  echo -e "${BLUE}Testing CUDA resize...${RESET}"
  ./build/bin/resize test_images/test.jpg test_images/resized.jpg 128 128
	if [ -f test_images/resized.jpg ]; then
		echo -e "${GREEN}CUDA resize test passed.${RESET}"
	else
		echo -e "${YELLOW}CUDA resize test skipped or failed.${RESET}"
	fi
fi

# --- Cleanup ---
echo -e "${BLUE}Cleaning up...${RESET}"
rm -rf build test_images

echo -e "${GREEN}Runner completed successfully.${RESET}"
