#!/bin/bash
# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.

GIT_ROOT=$(git rev-parse --show-toplevel)
TEST_DOCKER_DIR=$GIT_ROOT/docker/drum_integration_tests_base

IMAGE_NAME=datarobot/drum_integration_tests_base

echo "GIT_ROOT: $GIT_ROOT"
echo "TEST_DOCKER_DIR: $TEST_DOCKER_DIR"

pwd

cp -f $GIT_ROOT/custom_model_runner/requirements.txt $TEST_DOCKER_DIR/requirements_drum.txt
cp -f $GIT_ROOT/requirements.txt $TEST_DOCKER_DIR/requirements_dropin.txt


cd $TEST_DOCKER_DIR || exit 1
ls -la
echo
echo
echo "Building docker image for drum tests"
docker build -t $IMAGE_NAME ./

# Image is ready at this moment, but run make on drum to pull in all the java deps and commit the image
docker run -t -v $GIT_ROOT/custom_model_runner:/tmp/drum ${IMAGE_NAME} bash -c ". /opt/v3.7/bin/activate && cd /tmp/drum && make"
docker commit `docker ps -lq` $IMAGE_NAME
