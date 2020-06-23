#!/bin/bash

GIT_ROOT=$(git rev-parse --show-toplevel)
TEST_DOCKER_DIR=$GIT_ROOT/docker/drum_integration_tests_base

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
docker build -t datarobot/drum_integration_tests_base ./

