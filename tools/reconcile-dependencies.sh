#!/bin/bash

# This script generates fully pinned dependency lists for all environments in
# the public_dropin_environments and public_dropin_gpu_environments directories.
# It uses pip-tools' pip-compile to process requirements.in files and produce
# requirements.txt files with fully resolved dependencies.

set -e  # Exit immediately if a command exits with a non-zero status

# Ensure a specific pip version for compatibility
pip install pip==23.0.0

# Install or upgrade pip-tools for dependency management
pip install --upgrade pip-tools

# Define the directories containing the environments
env_dirs=("../public_dropin_gpu_environments" "../public_dropin_environments")

# Iterate over each environment directory
for dir in "${env_dirs[@]}"; do
    if [[ -d "$dir" ]]; then  # Check if the directory exists
        for env in "$dir"/*; do
            if [[ -d "$env" && -f "$env/requirements.in" ]]; then  # Check for a valid environment with requirements.in
                echo "Processing $env"
                # Generate a fully pinned requirements.txt file from requirements.in
                pip-compile --index-url=https://pypi.org/simple \
                            --no-annotate \
                            --no-emit-index-url \
                            --no-emit-trusted-host \
                            --verbose "$env/requirements.in"
            else
                echo "Skipping $env (no requirements.in found)"
            fi
            echo "----------------------------------------"
        done
    else
        echo "Skipping $dir (directory not found)"
    fi
done

echo "Dependency update complete!"
