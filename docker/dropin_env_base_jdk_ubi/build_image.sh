#!/usr/bin/env bash
# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_NAME_DATAROBOTDEV=datarobotdev/dropin-env-base-jdk
IMAGE_NAME_DATAROBOT=datarobot/dropin-env-base-jdk
IMAGE_TAG=ubi8.8-py3.11-jdk11.0.22-drum1.11.5-mlops9.2.8

pwd

echo "Building docker image: ${IMAGE_NAME_DATAROBOTDEV}:${IMAGE_TAG}"
export DATAROBOT_MLOPS_VERSION=9.2.8
${SCRIPT_DIR}/pull_artifacts.sh

# this is just a regular command to build an image for the host platform
docker build --build-arg DATAROBOT_MLOPS_VERSION=${DATAROBOT_MLOPS_VERSION} -t ${IMAGE_NAME_DATAROBOTDEV}:${IMAGE_TAG} -t ${IMAGE_NAME_DATAROBOT}:${IMAGE_TAG} . --push
docker push ${IMAGE_NAME_DATAROBOTDEV}:${IMAGE_TAG}
docker push ${IMAGE_NAME_DATAROBOT}:${IMAGE_TAG}

# this is command to build images for specified platforms. For more info: https://docs.docker.com/build/building/multi-platform/
#docker buildx build --build-arg DATAROBOT_MLOPS_VERSION=${DATAROBOT_MLOPS_VERSION} --platform linux/amd64 --push -t ${IMAGE_NAME_DATAROBOTDEV}:${IMAGE_TAG} -t ${IMAGE_NAME_DATAROBOT}:${IMAGE_TAG} .
