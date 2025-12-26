# CI/CD Pipeline

This document describes the continuous integration and deployment (CI/CD) pipeline used for the Hybrid Compute project, powered by CircleCI.

## Overview

The pipeline automates testing, building, and deployment of the project across multiple platforms and architectures. It ensures code quality, runs comprehensive tests, and deploys documentation.

## Workflows

### Main CI Workflow

Triggered on pushes to the `main` branch (excluding nightly runs).

- **Code Quality**: Lints code, runs pre-commit hooks, and performs static analysis with CodeQL.
- **Python Tests**: Runs tests across multiple Python versions (3.9-3.12).
- **Build Linux CUDA**: Builds and tests on Linux with CUDA support.
- **Build Linux CPU**: Builds and tests on Linux CPU-only.
- **Build Windows**: Builds on Windows with OpenCV.
- **Code Coverage**: Generates and uploads coverage reports.
- **Benchmark**: Runs performance benchmarks on macOS.
- **Docker Build**: Builds and pushes Docker images for CPU and CUDA variants.
- **Security Scan**: Scans Docker images for vulnerabilities.

### Nightly Workflow

Runs daily at 2 AM UTC on the main branch.

Includes all jobs from the main CI workflow for comprehensive nightly testing.

## Supported Platforms

- **Linux**: CUDA and CPU builds with GCC/Clang
- **macOS**: Metal builds with Xcode
- **Windows**: CPU builds with MSVC
- **Docker**: Multi-architecture images for CPU and GPU

## Key Features

- **Branch Filtering**: CI runs only on the main branch to avoid unnecessary builds on documentation branches.
- **Caching**: Build dependencies and artifacts are cached for faster subsequent runs.
- **Artifacts**: Test results, build outputs, and coverage reports are stored.
- **Security**: Automated vulnerability scanning of container images.
- **Documentation Deployment**: MkDocs site is automatically deployed to GitHub Pages.

## Configuration

The pipeline is configured in `.circleci/config.yml` with the following key components:

- **Orbs**: Reusable packages for Python, Docker, Node.js, AWS CLI, and Kubernetes.
- **Executors**: Pre-configured environments for different platforms.
- **Commands**: Reusable steps for dependency installation and building.
- **Jobs**: Specific tasks like testing, building, and deployment.
- **Workflows**: Orchestration of jobs with dependencies and triggers.

## Getting Started

To run the pipeline locally or contribute changes:

1. Ensure your branch is based on `main`.
2. Push changes to trigger the CI pipeline.
3. Monitor the CircleCI dashboard for build status.
4. Address any failing checks before merging.

For more details, see the `.circleci/config.yml` file in the repository.
