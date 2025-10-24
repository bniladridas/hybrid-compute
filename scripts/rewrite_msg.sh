#!/bin/bash

# Script to rewrite commit messages for history cleanup
# This script can be used with git filter-branch --msg-filter
# It enforces lowercase first line and truncates to 60 characters

# Read the commit message from stdin
message=$(cat)

# Get the first line
first_line=$(echo "$message" | head -n1)

# Convert to lowercase
first_line=$(echo "$first_line" | tr '[:upper:]' '[:lower:]')

# Truncate to 60 characters if longer
if [ ${#first_line} -gt 60 ]; then
	first_line=${first_line:0:60}
fi

# Output the rewritten message
echo "$first_line"
echo "$message" | tail -n +2
