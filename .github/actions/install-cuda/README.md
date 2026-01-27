# Install CUDA Action

This GitHub Action installs the CUDA toolkit and sets up the necessary environment variables for building CUDA applications.

## Features

- Supports multiple CUDA versions
- Automatic detection of Ubuntu version
- Configurable target architectures
- Optional dependency installation
- Comprehensive error handling and logging
- Environment variable setup for both shell and GitHub Actions

## Usage

### Basic Usage

```yaml
- name: Install CUDA
  uses: ./.github/actions/install-cuda
  with:
    cuda-version: '13.1'  # Optional, defaults to 13.1
    cuda-architectures: '75,80,86'  # Optional, defaults to '75,80,86'
    install-dependencies: 'true'  # Optional, defaults to true
```

### Advanced Usage

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install CUDA with specific version
        uses: ./.github/actions/install-cuda
        with:
          cuda-version: '13.1'
          cuda-architectures: '70,75,80,86,89,90'
          install-dependencies: 'true'

      - name: Build with CUDA
        run: |
          mkdir -p build && cd build
          cmake .. -DUSE_CUDA=ON
          make -j$(nproc)
```

## Inputs

| Input | Required | Default | Description |
|-------|----------|---------|-------------|
| `cuda-version` | No | '13.1' | Version of CUDA to install (e.g., '13.1') |
| `cuda-architectures` | No | '75,80,86' | Comma-separated list of CUDA architectures to target |
| `install-dependencies` | No | 'true' | Whether to install build dependencies |

## Outputs

This action sets the following environment variables:

- `CUDA_HOME`: Path to the CUDA installation directory
- `PATH`: Updated to include CUDA binaries
- `LD_LIBRARY_PATH`: Updated to include CUDA libraries
- `CUDACXX`: Path to the nvcc compiler
- `CMAKE_CUDA_ARCHITECTURES`: Semicolon-separated list of target architectures

## Supported Platforms

- Ubuntu 20.04 LTS (Focal Fossa)
- Ubuntu 22.04 LTS (Jammy Jellyfish)
- Other versions may work but are not officially supported

## License

This project is licensed under the [Apache License 2.0](LICENSE).
