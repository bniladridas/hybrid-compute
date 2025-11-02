# CircleCI

## CircleCI Status

| Component | Status | Details |
|-----------|--------|---------|
| Config validation | ✓ | YAML valid, orbs resolved |
| Multi-platform builds | ✓ | Linux CUDA, macOS Metal, Windows |
| Python matrix | ✓ | 3.9-3.12 testing |
| GPU resource allocation | ✓ | CUDA architectures 75;80;86 |
| Container builds | ✓ | CPU/GPU Docker images |
| Security scanning | ✓ | Trivy vulnerability checks |
| Benchmark execution | ✓ | Metal shim performance tests |
| Release automation | ✓ | Release-please integration |

## Workflows

### ci
Triggers: push/PR to main
- Code quality
- Python 3.9-3.12 matrix
- Linux CUDA/macOS Metal/Windows builds
- Coverage + security scans
- Docker CPU/GPU builds
- Release automation

### nightly
Triggers: daily 02:00 UTC
- Full test suite
- Extended benchmarks
- Security scans

## Environment Variables

```bash
GITHUB_TOKEN=<token>
DOCKER_USERNAME=<username>
DOCKER_PASSWORD=<password>
CODECOV_TOKEN=<token>
```

## Executors

| Executor | Image | Resources |
|----------|-------|-----------|
| ubuntu-cuda | nvidia/cuda:12.6.1-devel-ubuntu22.04 | gpu.nvidia.small |
| ubuntu-cpu | ubuntu:24.04 | large |
| macos-metal | Xcode 15.0 | macos.m1.medium.gen1 |
| windows-cuda | Windows Server 2022 | windows.medium |

## Validation

Validate the CircleCI configuration:

```bash
# Install CircleCI CLI (if not installed)
curl -fLSs https://raw.githubusercontent.com/CircleCI-Public/circleci-cli/master/install.sh | bash

# Validate configuration
circleci config validate .circleci/config.yml
```

## Dependencies

```
code-quality
├── python-tests
├── build-linux-cuda
├── build-linux-cpu
├── build-macos-metal
└── build-windows-cuda
    ├── code-coverage
    └── benchmark
        ├── docker-build
        └── security-scan
            └── release
```

## Codecov Integration

Both CI systems use identical Codecov v5 setup:

| CI System | Method | Token |
|-----------|--------|-------|
| GitHub Actions | `codecov-action@v5` | `${{ secrets.CODECOV_TOKEN }}` |
| CircleCI | `npx codecov@5` | `$CODECOV_TOKEN` |

## Artifacts

- Test results (JUnit)
- Coverage reports (Codecov)
- Build binaries
- Security scans (SARIF)
- Benchmarks

## Customization

### Python Versions
```yaml
python-tests:
  matrix:
    parameters:
      python-version: ["3.9", "3.10", "3.11", "3.12"]
```

### CUDA Architectures
```bash
-DCMAKE_CUDA_ARCHITECTURES=75;80;86
```

### New Platforms
1. Add executor
2. Create build job
3. Update workflow
