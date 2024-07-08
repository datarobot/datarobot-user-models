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

IMAGE_REPO=dropin-env-base-r

IMAGE_TAG=ubuntu20.04-r4.2.1-py3.8-jre11-drum1.11.5-mlops9.2.8-test

IMAGE_NAME_DATAROBOT=${IMAGE_ORG_DATAROBOT}/${IMAGE_REPO}:${IMAGE_TAG}
IMAGE_NAME_DATAROBOTDEV=${IMAGE_ORG_DATAROBOTDEV}/${IMAGE_REPO}:${IMAGE_TAG}

pwd

echo "Building docker image: ${IMAGE_NAME_DATAROBOTDEV}:${IMAGE_TAG}"

docker build -t ${IMAGE_NAME_DATAROBOT} -t ${IMAGE_NAME_DATAROBOTDEV} . --no-cache

if [ -n "${PUSH}" ] ; then
  docker push ${IMAGE_NAME_DATAROBOT}
  docker push ${IMAGE_NAME_DATAROBOTDEV}
fi
