<div align="center">
  <h1>🧸</h1>
</div>

**`Hybrid-compute`** is a cross-platform GPU-accelerated image processing framework with a CUDA-to-Metal compatibility shim. It enables CUDA-based operations on macOS (Apple Silicon) via Metal, supporting various image processing tasks including upscaling, filtering, color space conversion, morphology, thresholding, edge detection, and blending. Includes CPU preprocessing for tiling/stitching and utilities.
**Features**

- **Cross-Platform GPU Support**: CUDA-to-Metal shim for macOS, native CUDA on Linux/Windows, enabling GPU-accelerated image processing.
- **Image Processing Operations**: Supports upscaling, filtering, color space conversion, morphology, thresholding, edge detection, blending, and more.
- **Local Preprocessing**: CPU-based image tiling and stitching using OpenCV (C++) or stb_image (C) on macOS/Linux/Windows.
- **GPU Acceleration**: Optimized Metal shaders on macOS and CUDA kernels on Linux/Windows for high-performance processing.
- **Comprehensive Testing**: Includes unit tests, performance benchmarks, and end-to-end integration tests with parallel execution for faster CI/CD[^1].
- **CI/CD**: Automated builds and tests across macOS, Linux, and Windows with Docker image publishing and security scanning.
- **Notes**: Google Benchmark is enabled by default on macOS/Linux, but disabled on Windows due to linking issues (set in WindowsConfig.cmake). Modify ENABLE_BENCHMARK option in benchmark.cmake to adjust on other platforms. CI runs with parallel testing, non-interactive prompts, and containerized builds.
- **Development Documentation**: Detailed setup, architecture, onboarding, and compatibility guides[^2][^3].
  **Workflow**

1. **Split**: Process input images locally to create tiles.
2. Transfer and upscale tiles in the cloud.
3. Stitch upscaled tiles into the final image.
   **Prerequisites**

- macOS with Homebrew, Linux (Ubuntu) with apt, or Windows with Chocolatey
- CMake
- OpenCV (for C++ version) or stb_image (for C version)
- NumPy
- Python 3.10+ with pip
- Cloud instance with NVIDIA GPU and CUDA toolkit (for GPU upscaling)
  _Note: Conda is installed automatically on macOS via the setup script. The C version has no external dependencies beyond stb_image (header-only)._
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
brew install --cask miniforge
eval "$(\"$(brew --prefix)/Caskroom/miniforge/base/bin/conda\" shell.bash hook)"
conda init bash
mamba install -c conda-forge opencv cmake imagemagick -y
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
# Run benchmark test specifically (verbose output, tests Metal shim performance)
cd build && ctest -R user_counters_tabular_test -V
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
   _Note: Scripts are currently hardcoded for specific file names and grid sizes. Modify as needed for your use case._
   **Verification**
   To ensure the project components work correctly:

- **CUDA Build Check**: Run `scripts/check_cuda_build.sh` on a CUDA-enabled system to verify `upscale.cu` compiles without errors.
- **Local E2E Testing**: The `scripts/run.sh` script simulates the full pipeline (tiling → copy tiles → stitching) without actual upscaling or GPU hardware. `scripts/e2e.py` provides additional end-to-end validation.
- **Code Review**: Manually inspect `cloud_gpu/upscale.cu` for CUDA best practices and logic correctness.
- **Troubleshooting**: If you encounter build or test issues, see [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)[^4] for common problems and solutions.
  **Git Commit Standards**
This project enforces conventional commit standards for clean history:

### Commit Message Format
```
type(scope): short description (≤60 chars)
- optional bullet point 1 (≤72 chars)
- optional bullet point 2
```

### Rules
- **Type**: Must be one of:
  - `feat`: New feature
  - `fix`: Bug fix
  - `docs`: Documentation changes
  - `style`: Code style/formatting
  - `refactor`: Code change that neither fixes a bug nor adds a feature
  - `perf`: Performance improvements
  - `test`: Adding or modifying tests
  - `chore`: Maintenance tasks
  - `ci`: CI/CD related changes
  - `build`: Build system changes
  - `revert`: Revert a previous commit
- **Scope**: Lowercase with hyphens (e.g., `ci`, `api`, `ui`)
- **Description**: Short summary in lowercase (no period at the end)
- **Bullet Points**: Optional, each starting with `- ` and ≤72 characters
- **All text must be in lowercase**

### Examples
```
feat(api): add user authentication
- implement jwt token generation
- add login endpoint
ix(ci): resolve build failures
- update cmake minimum version
- fix opencv linking

docs(readme): update contribution guidelines
- add commit message format
- include code style requirements
```

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
Copyright (c) 2025, bniladridas. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions
are met:
  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the distribution.
  * Neither the name of bniladridas nor the names of its contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

---

This software links to the following components which are not licensed under the above license text.
For details on the specific licenses please refer to the provided links.

- OpenCV: https://opencv.org/license/
- stb_image: https://github.com/nothings/stb/blob/master/LICENSE

[^1]: [TESTING.md](https://github.com/bniladridas/hybrid-compute/blob/main/docs/TESTING.md) - Comprehensive testing procedures and guidelines.

[^2]: [DEVELOPMENT.md](https://github.com/bniladridas/hybrid-compute/blob/main/DEVELOPMENT.md) - Development setup, architecture, and contribution guidelines.

[^3]: [ONBOARDING.md](https://github.com/bniladridas/hybrid-compute/blob/main/docs/ONBOARDING.md) - Contributor onboarding and compatibility policy.

[^4]: [TROUBLESHOOTING.md](https://github.com/bniladridas/hybrid-compute/blob/main/docs/TROUBLESHOOTING.md) - Common issues and solutions for build and test problems.
