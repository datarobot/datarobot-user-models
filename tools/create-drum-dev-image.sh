#!/usr/bin/env bash
# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# 
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.

# This script can be used to create a dev image for testing 
# changes to drum within the DR app. Make sure to turn on
# the flag for uploading prebuilt images, and then upload
# the gzip file that this script outputs as a new execution
# environment.

set -ex

source "$(dirname "$0")/image-build-utils.sh"

build_drum

if [ -z "$1" ]; then
  echo "Please pass in a dropin env dir path to build"
  exit 1
fi

IMAGE_NAME="drum-testing-image-for-$1"

DRUM_WHEEL=$(find custom_model_runner/dist/datarobot_drum*.whl)
DRUM_WHEEL_REAL_PATH=$(realpath "$DRUM_WHEEL")

build_dropin_env_dockerfile "$1" "$DRUM_WHEEL_REAL_PATH"

pushd $1 || exit 1
docker build -t "$IMAGE_NAME" .
popd || exit 1

rm built-docker-image || true
docker save "$IMAGE_NAME" -o built-docker-image.tgz
echo "Your image has been saved to $(pwd)"
