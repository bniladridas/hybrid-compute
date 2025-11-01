#!/bin/bash

# Check if running in CI environment
if [[ -n "$CI" ]]; then
  # In CI, we want to fail if CircleCI config is invalid
  if ! command -v circleci &>/dev/null; then
    echo "::warning::CircleCI CLI not found in CI environment. Install with: brew install circleci"
    echo "Skipping CircleCI config validation"
    exit 0
  fi

  if ! circleci config validate; then
    echo "::error::CircleCI config validation failed"
    exit 1
  fi
else
  # Local development - just warn if CircleCI CLI is not available
  if command -v circleci &>/dev/null; then
    if ! circleci config validate; then
      echo "::warning::CircleCI config validation failed"
    fi
  else
    echo "::warning::CircleCI CLI not found. Install with: brew install circleci"
    echo "Skipping CircleCI config validation"
  fi
fi
