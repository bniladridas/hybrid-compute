#!/bin/bash  
CLOUD_IP="YOUR_CLOUD_IP"  
CLOUD_USER="ubuntu"  
CLOUD_PROJECT_PATH="/home/ubuntu/HybridCompute"  

# Transfer tiles to cloud  
scp -r ./test_images/tiles $CLOUD_USER@$CLOUD_IP:$CLOUD_PROJECT_PATH/tiles  

# Run CUDA upscaler on cloud  
ssh $CLOUD_USER@$CLOUD_IP "cd $CLOUD_PROJECT_PATH && nvcc cloud_gpu/upscale.cu -o upscaler -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_imgcodecs"  

# Transfer results back  
scp -r $CLOUD_USER@$CLOUD_IP:$CLOUD_PROJECT_PATH/upscaled ./test_images/  