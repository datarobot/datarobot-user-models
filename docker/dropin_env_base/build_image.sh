#!/usr/bin/env bash
# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_NAME=datarobot/dropin-env-base
IMAGE_TAG=debian11-py3.9-jre11.0.16-drum1.9.10-mlops8.2.7

pwd

echo "Building docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
export DATAROBOT_MLOPS_VERSION=8.2.7
${SCRIPT_DIR}/pull_artifacts.sh
docker build --build-arg DATAROBOT_MLOPS_VERSION=${DATAROBOT_MLOPS_VERSION} -t ${IMAGE_NAME}:${IMAGE_TAG} .
