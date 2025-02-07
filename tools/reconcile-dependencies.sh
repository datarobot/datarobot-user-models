#!/bin/bash

set -e

pip install pip==23.0.0

pip install --upgrade pip-tools

# Define the directories to process
env_dirs=("../public_dropin_gpu_environments" "../public_dropin_environments")

# Iterate over each environment directory
for dir in "${env_dirs[@]}"; do
    if [[ -d "$dir" ]]; then
        for env in "$dir"/*; do
            if [[ -d "$env" && -f "$env/requirements.in" ]]; then
                echo "Processing $env"
                pip-compile --index-url=https://pypi.org/simple --no-annotate --no-emit-index-url --no-emit-trusted-host --verbose "$env/requirements.in"
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
