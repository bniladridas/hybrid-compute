#!/bin/bash

if command -v circleci &>/dev/null; then
  circleci config validate
else
  echo "CircleCI CLI not found. Install with: brew install circleci"
  exit 1
fi
