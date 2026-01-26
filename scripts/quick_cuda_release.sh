#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, bniladridas. All rights reserved.

# Quick script to create CUDA release for current version

set -e

# Extract current CUDA version
CUDA_VERSION=$(grep "ARG CUDA_VERSION=" Dockerfile.cuda | cut -d= -f2)
BRANCH_NAME="cuda-$CUDA_VERSION"
TAG_NAME="v1.0.0-cuda-$CUDA_VERSION"

echo "🚀 Creating release for CUDA $CUDA_VERSION"

# Create and push branch
git checkout -b "$BRANCH_NAME" 2>/dev/null || git checkout "$BRANCH_NAME"
git push -u origin "$BRANCH_NAME"

# Create and push tag
git tag -a "$TAG_NAME" -m "Release v1.0.0 with CUDA $CUDA_VERSION support"
git push origin "$TAG_NAME"

# Create GitHub release
gh release create "$TAG_NAME" \
    --title "v1.0.0 - CUDA $CUDA_VERSION" \
    --notes "Release v1.0.0 with CUDA $CUDA_VERSION support

**CUDA Version:** $CUDA_VERSION
**Ubuntu Version:** 24.04

**Features:**
- Cross-platform GPU-accelerated image processing
- CUDA-to-Metal compatibility shim for macOS
- Dynamic Python version support in Docker
- Parameterized CUDA/Ubuntu versions

**Docker Usage:**
\`\`\`bash
docker build -f Dockerfile.cuda -t hybrid-compute-cuda:$CUDA_VERSION .
\`\`\`"

git checkout main

echo "✅ Release created: https://github.com/$(gh repo view --json owner,name -q '.owner.login + "/" + .name')/releases/tag/$TAG_NAME"
