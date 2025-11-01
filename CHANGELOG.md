# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Issue #31: Optimize CUDA build process to prevent memory issues
- Memory-optimized build configurations for CI/CD pipelines
- Job pool management for controlled parallel compilation

### Changed
- Updated CircleCI and GitHub Actions workflows with memory optimizations
- Limited parallel jobs in build processes to prevent OOM errors
- Enhanced build output verbosity for better diagnostics
- Optimized CUDA compiler flags for better memory management

### Fixed
- Resolved "Killed" and "Segmentation fault" errors during CUDA compilation
- Fixed memory exhaustion issues in CI/CD pipelines
- Addressed parallel build race conditions in CI environments

## [0.2.0] - 2025-11-01

### Added
- Initial project setup with CUDA and Metal support
- Basic image processing capabilities
- CI/CD pipeline with GitHub Actions
- Documentation and testing infrastructure
