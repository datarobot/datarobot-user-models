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

cat > Dockerfile << 'EOF'
FROM rocker/tidyverse:4.4.3

ENV LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 DEBIAN_FRONTEND=noninteractive

# This makes print statements show up in the logs API
ENV PYTHONUNBUFFERED=1

COPY requirements.txt requirements.txt

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt-get install --no-install-recommends -y \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        python3-pip \
        libglpk-dev \
        && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN Rscript -e "install.packages('caret', Ncpus=4)" && \
    Rscript -e "install.packages('recipes', Ncpus=4)" && \
    Rscript -e "install.packages('glmnet', Ncpus=4)" && \
    Rscript -e "install.packages('plumber', Ncpus=4)" && \
    Rscript -e "install.packages('Rook', Ncpus=4)" && \
    Rscript -e "install.packages('rjson', Ncpus=4)" && \
    Rscript -e "install.packages('e1071', Ncpus=4)" && \
    Rscript -e 'library(caret); install.packages(unique(modelLookup()[modelLookup()$forReg, c(1)]), Ncpus=4)' && \
    Rscript -e 'library(caret); install.packages(unique(modelLookup()[modelLookup()$forClass, c(1)]), Ncpus=4)' && \
    rm -rf /tmp/downloaded_packages/ /tmp/*.rds

# required by rpy2 https://github.com/rpy2/rpy2/issues/874
ENV LD_LIBRARY_PATH=/usr/local/lib/R/lib:/usr/local/lib:/usr/lib/x86_64-linux-gnu:/usr/lib/jvm/java-11-openjdk-amd64/lib/server
ENV VIRTUAL_ENV=/opt/venv

RUN sh -c "python -m venv ${VIRTUAL_ENV} && \
    . ${VIRTUAL_ENV}/bin/activate && \
    python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt"

ENV PATH=${VIRTUAL_ENV}/bin:${PATH}
ENV HOME=/opt
ENV CODE_DIR=/opt/code
ENV ADDRESS=0.0.0.0:8080

# This makes print statements show up in the logs API
ENV WITH_ERROR_SERVER=1 PYTHONUNBUFFERED=1

COPY ./*.sh ${CODE_DIR}/
WORKDIR ${CODE_DIR}

ENTRYPOINT ["sh", "-c", "${CODE_DIR}/start_server.sh \"$@\"", "--"]
EOF

docker build --no-cache -t ${target_tag} .

# Print out the image tag for the user. Keep blank line before and after the message.
echo ""
GREEN='\033[32m'
NC='\033[0m'
echo -e "Local development image built successfully.\nImage tag: ${GREEN}${target_tag}${NC}"
echo ""


popd > /dev/null # temp_dir