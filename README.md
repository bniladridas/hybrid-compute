<div align="center">
  <h1>🧸</h1>
</div>

**`Hybrid-compute`** is an image upscaling tool that works around macOS CUDA limits. It does the tiling on your local CPU, then sends the tiles to cloud GPUs for the heavy lifting. The tiles get upscaled with CUDA-powered bicubic interpolation on NVIDIA GPUs and are stitched back together into a clean, high-res image.
**Features**
- **Local Tile Splitting**: Efficiently divides images into 64x64 pixel tiles using OpenCV on macOS.
- **Cloud GPU Upscaling**: Performs 2x bicubic upscaling on tiles using CUDA kernels optimized for NVIDIA GPUs.
- **Local Stitching**: Recombines upscaled tiles into the final high-resolution image using Python and OpenCV.
**Workflow**
1. **Split**: Process input images locally to create tiles.  
2. Transfer and upscale tiles in the cloud.  
3. Stitch upscaled tiles into the final image.  
**Prerequisites**
- macOS with Homebrew or conda  
- CMake  
- OpenCV  
- NumPy  
- Python 3 with pip  
- Cloud instance with NVIDIA GPU and CUDA toolkit  
**Setup**
**Quick Setup**  
Run the setup script to install all dependencies:  
```bash
./scripts/setup.sh
```

**Docker Setup**  
For containerized environments:  
- **Local CPU components**:  
  ```bash
  docker build -t hybrid-compute .
  docker run --rm hybrid-compute
  ```
- **CUDA GPU components** (requires NVIDIA GPU):  
  ```bash
  docker build -f Dockerfile.cuda -t hybrid-compute-cuda .
  docker run --rm --gpus all -v /path/to/tiles:/app/tiles hybrid-compute-cuda ./cloud_gpu/upscaler tiles/input_tile.jpg tiles/output_tile.jpg
  ```
**Manual Setup**
**Local (macOS)**
```bash
# Install conda
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh && bash Miniconda3-latest-MacOSX-arm64.sh -b
source ~/miniconda3/bin/activate
conda install -c conda-forge opencv cmake imagemagick
# Clone repository
git clone https://github.com/bniladridas/hybrid-compute.git
cd hybrid-compute
# Install Python dependencies
pip install -r requirements.txt --user
python3 -m pip install opencv-python --user
# Test cv2 import
python3 -c "import cv2; print('cv2 works:', cv2.__version__)"
# Build local tools
mkdir build && cd build
cmake ..
make -j4
```
**Ubuntu**
```bash
sudo apt-get update
sudo apt-get install -y cmake libopencv-dev build-essential imagemagick
pip install -r requirements.txt
python3 -m pip install opencv-python --force
```
**Cloud GPU**
```bash
# On cloud instance with CUDA
cd cloud_gpu
nvcc upscale.cu -o upscaler -lopencv_core -lopencv_imgcodecs
```
**Usage**
**Quick Run**  
To build, test, and run e2e locally:
```bash
./scripts/run.sh
```
**Testing**
To run unit tests:
```bash
# Python tests
python3 -m pytest tests/
# C++ tests (after building)
cd build && ctest
```
**Manual Usage**
1. **Split images into tiles**:
   ```bash
   ./preprocess path/to/input_images/ path/to/tiles/
   ```
2. **Transfer tiles to cloud** (update script with your cloud details):
   ```bash
   ./scripts/transfer_tiles.sh
   ```
3. **Upscale tiles on cloud**:
    ```bash
    cd cloud_gpu && ./upscaler input_tile.jpg output_tile.jpg
    ```
    (Replace with actual tile filenames; defaults to input_tile.jpg/output_tile.jpg if no args provided)
4. **Stitch upscaled tiles** (currently hardcoded for 4x4 grid):
   ```bash
   python3 scripts/stitch.py
   ```
*Note: Scripts are currently hardcoded for specific file names and grid sizes. Modify as needed for your use case.*
**Verification**
To ensure the project components work correctly:
- **CUDA Build Check**: Run `scripts/check_cuda_build.sh` on a CUDA-enabled system to verify `upscale.cu` compiles without errors.
- **Local E2E Testing**: The `scripts/e2e.py` script now includes mock CPU-based upscaling to simulate the full pipeline (tiling → upscale → stitching) without requiring GPU hardware.
- **Code Review**: Manually inspect `cloud_gpu/upscale.cu` for CUDA best practices and logic correctness.
**License**  
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
