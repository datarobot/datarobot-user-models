#!/usr/bin/env bash
# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.

while [[ $# -gt 0 ]]; do
  case $1 in
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

IMAGE_TAG=debian11-py3.9-jre11.0.16-drum1.11.5-mlops9.2.8-test

IMAGE_NAME_DATAROBOT=${IMAGE_ORG_DATAROBOT}/${IMAGE_REPO}:${IMAGE_TAG}
IMAGE_NAME_DATAROBOTDEV=${IMAGE_ORG_DATAROBOTDEV}/${IMAGE_REPO}:${IMAGE_TAG}

pwd

echo "Building docker image: ${IMAGE_NAME_DATAROBOTDEV}:${IMAGE_TAG}"
export DATAROBOT_MLOPS_VERSION=9.2.8
${SCRIPT_DIR}/pull_artifacts.sh



# Build and save in the local registry. (In the harness pipeline we run trivy on it)
docker build --no-cache --build-arg DATAROBOT_MLOPS_VERSION=${DATAROBOT_MLOPS_VERSION} -t ${IMAGE_NAME_DATAROBOTDEV} -t ${IMAGE_NAME_DATAROBOT} .

# this is command to build images for specified platforms.
# For more info: https://docs.docker.com/build/building/multi-platform/
if [ -n "${PUSH}" ] ; then
  # When building for multiplatform, multiplatform manifest can not be saved locally
  # So need to build and push at the same time.
  docker buildx build --push --build-arg DATAROBOT_MLOPS_VERSION=${DATAROBOT_MLOPS_VERSION} --platform linux/amd64,linux/arm64 -t ${IMAGE_NAME_DATAROBOT} -t ${IMAGE_NAME_DATAROBOTDEV} .
fi