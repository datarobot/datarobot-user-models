#!/usr/bin/env bash
# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_NAME_DATAROBOTDEV=datarobotdev/dropin-env-base-r
IMAGE_NAME_DATAROBOT=datarobot/dropin-env-base-r
IMAGE_TAG=ubuntu20.04-r4.2.1-py3.8-jre11-drum1.11.5-mlops9.2.8

pwd

echo "Building docker image: ${IMAGE_NAME_DATAROBOTDEV}:${IMAGE_TAG}"
docker build -t ${IMAGE_NAME_DATAROBOTDEV}:${IMAGE_TAG} -t ${IMAGE_NAME_DATAROBOT}:${IMAGE_TAG} . --no-cache --push
