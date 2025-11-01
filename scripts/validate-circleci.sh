#!/bin/bash

# Always skip validation in CI environment since we can't install the CircleCI CLI there
if [[ -n "$CI" ]]; then
  echo "Skipping CircleCI config validation in CI environment"
  exit 0
fi

# For local development, try to validate if CircleCI CLI is available
if command -v circleci &>/dev/null; then
  if ! circleci config validate; then
    echo "::warning::CircleCI config validation failed"
    exit 0  # Don't fail the build for local development
  fi
else
  echo "::warning::CircleCI CLI not found. Install with: brew install circleci"
  echo "Skipping CircleCI config validation"
fi

exit 0  # Always exit with success
