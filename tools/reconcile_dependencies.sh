#!/bin/bash

# This script generates fully pinned dependency lists for an environment.
# It uses pip-tools' pip-compile to process requirements.in file and produce
# requirements.txt file with fully resolved dependencies.

set -e  # Exit immediately if a command exits with a non-zero status

# Ensure a specific pip version for compatibility
pip install pip==23.0.0

# Install or upgrade pip-tools for dependency management
pip install --upgrade pip-tools

# Set the RPY2_CFFI_MODE environment variable to ABI to avoid issues with rpy2
export RPY2_CFFI_MODE=ABI

if [[ -f "requirements.in" ]]; then  # Check if requirements.in exists
    # Generate a fully pinned requirements.txt file from requirements.in
    pip-compile --index-url=https://pypi.org/simple \
                --no-annotate \
                --no-emit-index-url \
                --no-emit-trusted-host \
                --verbose "requirements.in"
else
    echo "Skipping (no requirements.in found)"
fi

echo "Dependency update complete!"