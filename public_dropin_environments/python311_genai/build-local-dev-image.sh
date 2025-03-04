#!/usr/bin/env bash
# This script builds a local development image for Python 3.11, similar to the production image.
# Both images are based on Alpine Linux and use the same Python version. However, the production image is built on
# a Chainguard base image, while the local development image is based on the official Python image.

set -e

BASE_IMAGE="python:3.11-slim-bookworm"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
parent_dir=$(basename "${script_dir}")
target_tag="${parent_dir//_/-}-local-dev"

# Copy the current folder content into a temporary folder
temp_dir=$(mktemp -d)
cp -r ${script_dir}/* "${temp_dir}/"

pushd "${temp_dir}" || exit 1

cat > Dockerfile <<EOF
FROM ${BASE_IMAGE}

COPY requirements.txt requirements.txt

ENV VIRTUAL_ENV=/opt/venv

RUN sh -c "python -m venv \${VIRTUAL_ENV} && \
    . \${VIRTUAL_ENV}/bin/activate && \
    python -m ensurepip --default-pip && \
    python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt"

ENV PATH=\${VIRTUAL_ENV}/bin:\${PATH}
ENV HOME=/opt
ENV CODE_DIR=/opt/code
ENV ADDRESS=0.0.0.0:8080

# This makes print statements show up in the logs API
ENV WITH_ERROR_SERVER=1 PYTHONUNBUFFERED=1

COPY ./*.sh \${CODE_DIR}/
WORKDIR \${CODE_DIR}

ENTRYPOINT ["sh", "-c", "\${CODE_DIR}/start_server.sh \"\$@\"", "--"]
EOF

docker build -t ${target_tag} .

# Print out the image tag for the user. Keep blank line before and after the message.
echo ""
GREEN='\033[32m'
NC='\033[0m'
echo -e "Local development image built successfully.\nImage tag: ${GREEN}${target_tag}${NC}"
echo ""


popd > /dev/null # temp_dir