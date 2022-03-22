#!/usr/bin/env bash
# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_NAME=datarobot/dropin-env-base
IMAGE_TAG=debian10-py3.7-jre11-drum1.8.0-mlops8.0.7

pwd

echo "Building docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
DATAROBOT_MLOPS_VERSION=8.0.7 ${SCRIPT_DIR}/pull_artifacts.sh
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
