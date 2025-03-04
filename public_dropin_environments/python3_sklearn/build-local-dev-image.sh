#!/usr/bin/env bash
# This script builds a local development image for Python 3.11, similar to the production image.
# Both images are based on Alpine Linux and use the same Python version. However, the production image is built on
# a Chainguard base image, while the local development image is based on the official Python image.

set -e

BASE_IMAGE="python:3.11-alpine"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
parent_dir=$(basename "${script_dir}")
target_tag="${parent_dir//_/-}-local-dev"

# Copy the current folder content into a temporary folder
temp_dir=$(mktemp -d)
cp -r ${script_dir}/* "${temp_dir}/"

pushd "${temp_dir}" || exit 1

sed -i 's/# MARK: LOCAL-DEV-BUILD-ADD-HERE./RUN apk add --no-cache build-base python3-dev linux-headers/' "${temp_dir}/Dockerfile"

# Replace the '/usr/bin/pip' in the Dockerfile with '/usr/local/bin/pip'
sed -i 's/\/usr\/bin\/pip/\/usr\/local\/bin\/pip/g' "${temp_dir}/Dockerfile"

docker build \
    --build-arg PRODUCTION_BASE_IMAGE=${BASE_IMAGE} \
    --build-arg DEVELOPMENT_BASE_IMAGE=${BASE_IMAGE} \
    -t ${target_tag} \
    .

# Print out the image tag for the user. Keep blank line before and after the message.
echo ""
GREEN='\033[32m'
NC='\033[0m'
echo -e "Local development image built successfully.\nImage tag: ${GREEN}${target_tag}${NC}"
echo ""


popd > /dev/null # temp_dir