# Development Setup Guide

This guide provides instructions for setting up the development environment for the Hybrid Compute project.

## Prerequisites

### macOS Requirements
1. **Xcode** (Latest version from Mac App Store)
   - Install from [Mac App Store](https://apps.apple.com/us/app/xcode/id497799835)
   - Or download from [Apple Developer](https://developer.apple.com/download/)

2. **Command Line Tools**
   ```bash
   xcode-select --install
   ```

3. **Homebrew** (Package Manager)
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

## Setup Instructions

### 1. Install Dependencies

```bash
# Install CMake and build tools
brew install cmake ninja

# Install Python (if not already installed)
brew install python@3.9
```

### 2. Configure Xcode

1. Open Xcode at least once to accept the license agreement
2. Set the command line tools:
   ```bash
   sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
   sudo xcodebuild -license accept
   ```

### 3. Build the Project

```bash
# Clone the repository (if not already cloned)
git clone <repository-url>
cd hybrid-compute

# Create build directory and configure
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=OFF

# Build the project
make -j$(sysctl -n hw.logicalcpu)
```

## Common Issues and Solutions

### 1. Metal Compiler Not Found

**Error**: `error: cannot execute tool 'metal' due to missing Metal Toolchain`

**Solution**:
1. Ensure Xcode is properly installed and opened at least once
2. Run:
   ```bash
   sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
   sudo xcodebuild -license accept
   ```
3. If the issue persists, reinstall Xcode Command Line Tools:
   ```bash
   sudo rm -rf /Library/Developer/CommandLineTools
   xcode-select --install
   ```

### 2. CMake Configuration Errors

**Error**: `CMake Error at CMakeLists.txt`

**Solution**:
1. Ensure you have the required CMake version (3.10+)
   ```bash
   brew upgrade cmake
   ```
2. Clean and reconfigure:
   ```bash
   rm -rf build
   mkdir -p build && cd build
   cmake ..
   ```

### 3. Build Failures

**Error**: Build fails with linker errors

**Solution**:
1. Clean the build directory:
   ```bash
   rm -rf build/*
   ```
2. Rebuild from scratch:
   ```bash
   mkdir -p build && cd build
   cmake ..
   make clean
   make -j$(sysctl -n hw.logicalcpu)
   ```

## Development Workflow

### Running Tests

```bash
cd build
ctest --output-on-failure
```

### Running Benchmarks

```bash
./tests/benchmark_metal_shim
```

### Code Formatting

We use `clang-format` for code formatting:

```bash
# Install clang-format
brew install clang-format

# Format all source files
find src include tests -name "*.h" -o -name "*.cpp" -o -name "*.hpp" | xargs clang-format -i
```

## IDE Setup

### Visual Studio Code

Recommended extensions:
- C/C++
- CMake
- CMake Tools
- clangd

### Xcode

You can generate an Xcode project with:
```bash
mkdir -p build-xcode && cd build-xcode
cmake -G Xcode ..
open HybridCompute.xcodeproj
```

## Troubleshooting

### Missing Dependencies

If you encounter missing dependencies, you can install them using Homebrew:

```bash
# Example: Installing OpenMP
brew install libomp
```

### Permission Issues

If you encounter permission issues, try:

```bash
# Fix permissions for build directory
sudo chown -R $(whoami) build/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
