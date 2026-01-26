#!/bin/bash

# Batch upscaler script for processing multiple tiles
# Usage: ./batch_upscale.sh <input_dir> <output_dir> [scale] [pattern]

set -e

INPUT_DIR="${1:-tiles}"
OUTPUT_DIR="${2:-upscaled}"
SCALE="${3:-2}"
PATTERN="${4:-*.jpg}"
UPSCALER="${UPSCALER:-./upscaler}"

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

if [ ! -f "$UPSCALER" ]; then
    echo "Error: Upscaler binary '$UPSCALER' not found"
    echo "Build it first with: nvcc cloud_gpu/upscale.cu -o upscaler -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -std=c++17"
    exit 1
fi

echo "Processing tiles from $INPUT_DIR to $OUTPUT_DIR with scale $SCALE"
echo "Pattern: $PATTERN"

count=0
for file in "$INPUT_DIR"/$PATTERN; do
    if [ -f "$file" ]; then
        basename=$(basename "$file")
        output_file="$OUTPUT_DIR/$basename"
        echo "Processing: $basename"
        "$UPSCALER" "$file" "$output_file" "$SCALE"
        ((count++))
    fi
done

echo "Processed $count files"
