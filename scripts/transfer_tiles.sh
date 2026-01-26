#!/bin/bash

# Configuration - update these values for your setup
CLOUD_IP="${CLOUD_IP:-YOUR_CLOUD_IP}"
CLOUD_USER="${CLOUD_USER:-ubuntu}"
CLOUD_PROJECT_PATH="${CLOUD_PROJECT_PATH:-/home/ubuntu/HybridCompute}"
LOCAL_TILES_DIR="${LOCAL_TILES_DIR:-./test_images/tiles}"
REMOTE_TILES_DIR="${REMOTE_TILES_DIR:-tiles}"
UPSCALER_NAME="${UPSCALER_NAME:-upscaler}"

if [ "$CLOUD_IP" = "YOUR_CLOUD_IP" ]; then
    echo "Error: Please set CLOUD_IP environment variable or update the script"
    echo "Usage: CLOUD_IP=x.x.x.x $0"
    exit 1
fi

echo "Transferring tiles from $LOCAL_TILES_DIR to $CLOUD_USER@$CLOUD_IP:$CLOUD_PROJECT_PATH/$REMOTE_TILES_DIR"

# Transfer tiles to cloud
scp -r "$LOCAL_TILES_DIR" "$CLOUD_USER@$CLOUD_IP:$CLOUD_PROJECT_PATH/$REMOTE_TILES_DIR"

echo "Building CUDA upscaler on cloud..."
# Build CUDA upscaler on cloud
ssh "$CLOUD_USER@$CLOUD_IP" "cd $CLOUD_PROJECT_PATH && nvcc cloud_gpu/upscale.cu -o $UPSCALER_NAME -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -std=c++17"

echo "Processing complete. Transfer results back with:"
echo "scp -r $CLOUD_USER@$CLOUD_IP:$CLOUD_PROJECT_PATH/upscaled ./test_images/"
