#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, bniladridas. All rights reserved.

set -e

# Function to extract CUDA version from Dockerfile
get_cuda_version() {
    local commit=$1
    git show "$commit:Dockerfile.cuda" 2>/dev/null | grep -o 'nvidia/cuda:[0-9]\+\.[0-9]\+\.[0-9]\+' | head -1 | cut -d: -f2 || echo ""
}

# Function to create branch, tag, and release
create_cuda_release() {
    local commit=$1
    local cuda_version=$2
    local branch_name="cuda-$cuda_version"
    local tag_name="v1.0.0-cuda-$cuda_version"
    
    echo "Processing CUDA $cuda_version from commit $commit"
    
    # Check if branch already exists
    if git show-ref --verify --quiet "refs/heads/$branch_name"; then
        echo "Branch $branch_name already exists, skipping..."
        return
    fi
    
    # Create branch from commit
    git checkout "$commit" -b "$branch_name" 2>/dev/null || {
        git checkout "$branch_name"
    }
    
    # Push branch
    git push -u origin "$branch_name" || echo "Branch already pushed"
    
    # Create tag
    git tag -a "$tag_name" -m "Release v1.0.0 with CUDA $cuda_version support" || echo "Tag already exists"
    
    # Push tag
    git push origin "$tag_name" || echo "Tag already pushed"
    
    # Create GitHub release
    gh release create "$tag_name" \
        --title "v1.0.0 - CUDA $cuda_version" \
        --notes "Release v1.0.0 with CUDA $cuda_version support

**CUDA Version:** $cuda_version
**Ubuntu Version:** 24.04

**Docker Usage:**
\`\`\`bash
docker build -f Dockerfile.cuda -t hybrid-compute-cuda:$cuda_version .
\`\`\`" || echo "Release already exists"
}

# Main script
echo "🚀 Automating CUDA version releases..."

# Get commits that modified Dockerfile.cuda
commits=$(git log --oneline --follow Dockerfile.cuda | awk '{print $1}')

declare -A seen_versions

for commit in $commits; do
    cuda_version=$(get_cuda_version "$commit")
    
    if [[ -n "$cuda_version" && -z "${seen_versions[$cuda_version]}" ]]; then
        seen_versions[$cuda_version]=1
        create_cuda_release "$commit" "$cuda_version"
    fi
done

# Return to main branch
git checkout main

echo "✅ CUDA release automation complete!"
echo "📋 Created releases for versions: ${!seen_versions[*]}"
