#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, bniladridas. All rights reserved.

set -euo pipefail

VERSION_VALUE=$(tr -d '[:space:]' < VERSION)

if [[ ! "$VERSION_VALUE" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "[ERROR] VERSION must contain exactly one semantic version. Found: ${VERSION_VALUE}" >&2
    exit 1
fi

if rg -n "release-please" .github scripts . -g '!CHANGELOG.md' -g '!scripts/validate_release_config.sh' >/dev/null 2>&1; then
    echo "[ERROR] Found stale release-please references outside CHANGELOG.md" >&2
    rg -n "release-please" .github scripts . -g '!CHANGELOG.md' -g '!scripts/validate_release_config.sh'
    exit 1
fi

if rg -n "v1\\.0\\.0-cuda-|Release v1\\.0\\.0 with CUDA|v1\\.0\\.0 - CUDA" scripts .github >/dev/null 2>&1; then
    echo "[ERROR] Found hardcoded CUDA release version strings" >&2
    rg -n "v1\\.0\\.0-cuda-|Release v1\\.0\\.0 with CUDA|v1\\.0\\.0 - CUDA" scripts .github
    exit 1
fi

if [[ ! -x scripts/create_release.sh ]]; then
    echo "[ERROR] scripts/create_release.sh must be executable" >&2
    exit 1
fi

echo "[PASS] Release configuration is valid for VERSION=${VERSION_VALUE}"
