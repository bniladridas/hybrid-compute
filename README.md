<div align="center">
  <h1>🧸</h1>
</div>

**`Hybrid-compute`** is an image upscaling tool that leverages hybrid CPU/GPU computing for efficient image processing. It splits images into tiles locally, upscales them using isomorphic bicubic interpolation (GPU-accelerated with CUDA if available, CPU fallback with OpenMP), and stitches them back into high-resolution images.
**Features**
- **Local Tile Splitting**: Efficiently divides images into tiles using OpenCV.
- **Isomorphic Upscaling**: Performs 2x bicubic upscaling on tiles using CUDA on GPU or OpenMP on CPU, with automatic fallback.
- **Local Stitching**: Recombines upscaled tiles into the final high-resolution image using Python and OpenCV.
**Workflow**
1. **Split**: Process input images locally to create tiles.
2. **Upscale**: Upscale tiles locally using GPU (if available) or CPU.
3. **Stitch**: Combine upscaled tiles into the final image.  
**Prerequisites**
- macOS, Linux, or Windows
- CMake
- OpenCV
- NumPy
- Python 3 with pip
- Optional: NVIDIA GPU with CUDA toolkit for GPU acceleration  
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
  docker run --rm --gpus all hybrid-compute-cuda ./build/upscale input_tile.jpg output_tile.jpg
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
make -j4  # Builds preprocess, upscale (if CUDA available), and tests
```
**Ubuntu**
```bash
sudo apt-get update
sudo apt-get install -y cmake libopencv-dev build-essential imagemagick
pip install -r requirements.txt
python3 -m pip install opencv-python --force
```
**GPU Support**
The upscale tool automatically detects GPU availability and uses CUDA for acceleration, falling back to CPU with OpenMP if no GPU is found.
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
   ./build/preprocess path/to/input_images/ path/to/tiles/
   ```
2. **Upscale tiles**:
   ```bash
   ./build/upscale input_tile.jpg output_tile.jpg [scale]
   ```
   (Scale defaults to 2; uses GPU if available, CPU otherwise)
3. **Stitch upscaled tiles**:
   ```bash
   python3 scripts/stitch.py path/to/upscaled_tiles/ output_image.jpg
   ```
*Note: Adjust paths and grid sizes as needed for your use case.*
**Verification**
To ensure the project components work correctly:
- **Build Check**: Run `mkdir build && cd build && cmake .. && make` to verify all components compile.
- **Local E2E Testing**: The `scripts/e2e.py` script runs the full pipeline (tiling → upscale → stitching) with actual GPU/CPU upscaling.
- **Unit Tests**: Run `python3 -m pytest tests/` and `cd build && ctest` for comprehensive testing.
- **Code Review**: Inspect `cloud_gpu/upscale.cu` for isomorphic CUDA/CPU logic and `tests/test_upscaler.cpp` for unit tests.
**Git Commit Standards**
This project enforces conventional commit standards for clean history:
- Commit messages must start with a type: `feat:`, `fix:`, `docs:`, `style:`, `refactor:`, `perf:`, `test:`, `chore:`, `ci:`, `build:`, or `revert:`.
- All text must be lowercase.
- The first line must be ≤60 characters.
To enable enforcement, copy the hook:
```bash
cp scripts/commit-msg .git/hooks/commit-msg
chmod +x .git/hooks/commit-msg
```
To clean up existing commit messages in history:
```bash
git filter-branch --msg-filter 'bash scripts/rewrite_msg.sh' -- --all
git push --force origin main  # if needed
```
**License**  
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
