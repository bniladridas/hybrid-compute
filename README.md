<div align="center">
  <h1>🧸</h1>
</div>

**`Hybrid-compute`** is a cross-platform image upscaling tool that leverages GPU acceleration for high-performance image processing. It supports Metal on macOS and CUDA on Linux/Windows, providing efficient 2x bicubic upscaling with local CPU tiling and stitching.
**Features**
- **Cross-Platform GPU Support**: Uses Metal shaders on macOS and CUDA kernels on Linux/Windows for hardware-accelerated upscaling.
- **Local Tile Splitting**: Efficiently divides images into 64x64 pixel tiles using OpenCV (C++) or stb_image (C) on macOS/Linux/Windows.
- **GPU Upscaling**: Performs 2x bicubic interpolation on tiles using optimized GPU backends.
- **Local Stitching**: Recombines upscaled tiles into the final high-resolution image using Python and OpenCV.
- **Comprehensive Testing**: Includes unit tests, performance benchmarks, and end-to-end integration tests with parallel execution for faster CI/CD[^1].
- **Development Documentation**: Detailed setup and architecture guides[^2].
**Workflow**
1. **Split**: Process input images locally to create tiles.
2. Transfer and upscale tiles in the cloud.
3. Stitch upscaled tiles into the final image.
**Prerequisites**
- macOS with Homebrew, Linux (Ubuntu) with apt, or Windows with Chocolatey
- CMake
- OpenCV (for C++ version) or stb_image (for C version)
- NumPy
- Python 3.9+ with pip
- Cloud instance with NVIDIA GPU and CUDA toolkit (for GPU upscaling)
*Note: Conda is installed automatically on macOS via the setup script. The C version has no external dependencies beyond stb_image (header-only).*
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
**macOS**
```bash
# Install dependencies
brew install --cask miniconda
eval "$(/opt/homebrew/Caskroom/miniconda/base/bin/conda shell.bash hook)"
conda init bash
conda install -c conda-forge opencv cmake imagemagick -y
# Install Python dependencies
pip install -r requirements.txt --user
# Test cv2 import
python3 -c "import cv2; print('cv2 works:', cv2.__version__)"
# Build local tools
mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.logicalcpu)
```
**Ubuntu**
```bash
sudo apt-get update
sudo apt-get install -y cmake libopencv-dev build-essential imagemagick wget
# Install CUDA (optional)
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
sudo dpkg -i /tmp/cuda-keyring.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-11-8 || echo "CUDA installation skipped"
pip install -r requirements.txt
```
**Windows**
```bash
# Install dependencies
choco install cmake opencv imagemagick -y
# Install Python dependencies
pip install -r requirements.txt --user
# Build local tools
mkdir build && cd build
cmake ..
cmake --build . --config Release
```
**Cloud GPU**
```bash
# On cloud instance with CUDA
cd cloud_gpu
nvcc upscale.cu -o upscaler -I/usr/include/opencv4 -lopencv_core -lopencv_imgcodecs
```
**Usage**
**Quick Run**
To build, test, and run e2e locally:
```bash
./scripts/run.sh
```
**Testing**
For detailed testing instructions, see TESTING.md[^1].

To run unit tests (parallel execution enabled for faster runs):
```bash
# Python tests (parallel with pytest-xdist)
python3 -m pytest tests/
# C/C++ tests (parallel with ctest)
cd build && ctest -j$(nproc)
# End-to-end tests
python3 scripts/e2e.py
```
**Manual Usage**
1. **Split images into tiles** (C++ version with OpenCV):
   ```bash
   ./preprocess path/to/input_images/ path/to/tiles/
   ```
   Or (C version with stb_image, no OpenCV required):
   ```bash
   ./preprocess_c path/to/input_images/ path/to/tiles/
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
   python3 scripts/stitch.py path/to/upscaled_tiles/ output_image.jpg
   ```
*Note: Scripts are currently hardcoded for specific file names and grid sizes. Modify as needed for your use case.*
**Verification**
To ensure the project components work correctly:
- **CUDA Build Check**: Run `scripts/check_cuda_build.sh` on a CUDA-enabled system to verify `upscale.cu` compiles without errors.
- **Local E2E Testing**: The `scripts/run.sh` script simulates the full pipeline (tiling → copy tiles → stitching) without actual upscaling or GPU hardware. `scripts/e2e.py` provides additional end-to-end validation.
- **Code Review**: Manually inspect `cloud_gpu/upscale.cu` for CUDA best practices and logic correctness.
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

[^1]: [TESTING.md](https://github.com/bniladridas/hybrid-compute/blob/main/docs/TESTING.md) - Comprehensive testing procedures and guidelines.
[^2]: [DEVELOPMENT.md](https://github.com/bniladridas/hybrid-compute/blob/main/DEVELOPMENT.md) - Development setup, architecture, and contribution guidelines.
