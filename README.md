# HybridCompute: A Solution for Advanced Image Processing  

This is **HybridCompute**, a system I developed to enhance image processing capabilities. Designed to overcome the lack of CUDA support on macOS, it combines local preprocessing with cloud-based GPU power for efficient, high-quality results.

---

## Overview  
- **Tile-Splitting**: Uses OpenCV on a Mac to divide images into manageable tiles, leveraging ARM NEON for optimized performance.  
- **GPU Upscaling**: Employs NVIDIA GPUs in the cloud to upscale tiles using CUDA with a bicubic interpolation method.  
- **Stitching**: Reassembles the processed tiles into a single, high-resolution image.  

This approach bypasses macOS’s CUDA limitation by splitting the workload between local and remote resources.

---

## Purpose  
I created HybridCompute to address the challenge of needing GPU acceleration while working on macOS. It preprocesses images locally on an M1 Mac and taps into cloud GPUs for upscaling, offering a practical solution for users in a similar situation.

---

## Workflow  
1. **Local Preprocessing**: The Mac splits the input image into tiles.  
2. **Cloud Processing**: Tiles are transferred to a cloud GPU for CUDA-based upscaling.  
3. **Final Assembly**: The upscaled tiles are stitched back into a complete image.

---

## Setup Instructions  

### Local Mac Configuration  
```bash  
# Install OpenCV  
brew install opencv  

# Clone the repository  
git clone https://github.com/bniladridas/HybridCompute.git  
cd HybridCompute  

# Build the project  
mkdir build && cd build  
cmake .. -DCMAKE_BUILD_TYPE=Release  
make -j4  
```

### Cloud GPU Configuration  
```bash  
# Navigate to the GPU directory  
cd cloud_gpu  

# Compile the CUDA upscaling code  
nvcc upscale.cu -o upscaler -lopencv_core -lopencv_imgcodecs  
```

---

## Usage  

### Split the Image  
```bash  
./preprocess my_image.jpg tiles/  
# This generates tiles in the specified directory.  
```

### Upscale the Tiles  
```bash  
# Transfer tiles to the cloud  
./scripts/transfer_tiles.sh  

# Process tiles on the GPU  
cd cloud_gpu  
./upscaler  
```

### Stitch the Result  
```bash  
python3 scripts/stitch.py --input upscaled/ --output my_masterpiece.jpg  
# Ensure inputs are valid for best results.  
```

---

## Performance  
| Machine         | 4K Processing Time | Notes               |  
|-----------------|--------------------|---------------------|  
| M1 Mac (CPU)    | 14.7s             | Reliable baseline   |  
| NVIDIA T4       | 2.3s              | Efficient scaling   |  
| NVIDIA A100     | 0.9s              | High-end option     |  

---

## Technical Details  
- **CUDA**: Utilizes 16x16 thread blocks for efficient memory management.  
- **Bicubic Interpolation**: Provides clean, high-quality upscaling.  
- **Error Handling**: Built-in checks to ensure smooth execution.

---

## Limitations  
- No CUDA support on macOS requires cloud reliance.  
- Bicubic upscaling performs best with high-quality inputs.  
- Cloud usage may incur costs depending on the provider.

---

## Future Plans  
- Explore Metal for local GPU acceleration on macOS.  
- Consider AI-based upscaling for enhanced results.  
- Optimize tile transfer speeds for better efficiency.

---

## Contribution  
Feel free to fork the project, experiment, and submit improvements via pull requests. Feedback is welcome to refine the system further.

---

## License  
Distributed under the MIT License. Use it freely, but note that cloud-related expenses are your responsibility.
