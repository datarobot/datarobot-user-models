#!/usr/bin/env bash
# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.

DATAROBOT_BASE_IMAGE_VERSION=1.0.0

while [[ $# -gt 0 ]]; do
  case $1 in
    --version)
      shift
      DATAROBOT_BASE_IMAGE_VERSION=$1
      shift # past argument
      ;;
    --push)
      PUSH=1
      shift # past argument
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_ORG_DATAROBOT=datarobot
IMAGE_ORG_DATAROBOTDEV=datarobotdev

IMAGE_REPO=dropin-env-base

BASE_ROOT_IMAGE_TAG=$(grep -E '^ARG[[:space:]]+BASE_ROOT_IMAGE=' Dockerfile | sed -E 's/^ARG[[:space:]]+BASE_ROOT_IMAGE=//')
BASE_ROOT_IMAGE_TAG="${BASE_ROOT_IMAGE_TAG//:/-}"
IMAGE_TAG=${DATAROBOT_BASE_IMAGE_VERSION}-${BASE_ROOT_IMAGE_TAG}

IMAGE_NAME_DATAROBOT=${IMAGE_ORG_DATAROBOT}/${IMAGE_REPO}:${IMAGE_TAG}
IMAGE_NAME_DATAROBOTDEV=${IMAGE_ORG_DATAROBOTDEV}/${IMAGE_REPO}:${IMAGE_TAG}

pwd

echo "Building docker image: ${IMAGE_NAME_DATAROBOTDEV}"

# Build and save in the local registry. (In the harness pipeline we run trivy on it)
docker build --no-cache -t ${IMAGE_NAME_DATAROBOTDEV} -t ${IMAGE_NAME_DATAROBOT} .

# this is command to build images for specified platforms.
# For more info: https://docs.docker.com/build/building/multi-platform/
if [ -n "${PUSH}" ] ; then
  # When building for multiplatform, multiplatform manifest can not be saved locally
  # So need to build and push at the same time.
  docker buildx build --push --build-arg DATAROBOT_MLOPS_VERSION=${DATAROBOT_MLOPS_VERSION} --platform linux/amd64,linux/arm64 -t ${IMAGE_NAME_DATAROBOT} -t ${IMAGE_NAME_DATAROBOTDEV} .
fi