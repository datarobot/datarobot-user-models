#!/bin/bash
# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# 
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.

GIT_ROOT=$(git rev-parse --show-toplevel)
BUILD_DOCKER_DIR=$GIT_ROOT/docker/drum_builder

IMAGE_NAME=datarobot/drum-builder

echo "GIT_ROOT: $GIT_ROOT"
echo "BUILD_DOCKER_DIR: $BUILD_DOCKER_DIR"

cp -f "$GIT_ROOT/custom_model_runner/requirements.txt" "$BUILD_DOCKER_DIR/requirements_drum.txt"


pushd "$BUILD_DOCKER_DIR" || exit 1
ls -la
echo
echo
echo "Building docker image for DRUM compilation"
docker build -t $IMAGE_NAME ./
popd || exit 1

# Image is ready at this moment, but run make on drum to pull in all the java deps and commit the image
docker run -t --user "$(id -u)":"$(id -g)" -v "$GIT_ROOT/custom_model_runner:/tmp/drum" ${IMAGE_NAME} bash -c "cd /tmp/drum && make"
docker commit "$(docker ps -lq)" $IMAGE_NAME
