#!/bin/bash
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

GIT_ROOT=$(git rev-parse --show-toplevel)
BUILD_DOCKER_DIR=$GIT_ROOT/docker/drum_builder

IMAGE_ORG_DATAROBOT=datarobot
IMAGE_ORG_DATAROBOTDEV=datarobotdev

IMAGE_REPO=drum-builder

IMAGE_TAG=ubuntu-22-04-test

IMAGE_NAME_DATAROBOT=${IMAGE_ORG_DATAROBOT}/${IMAGE_REPO}:${IMAGE_TAG}
IMAGE_NAME_DATAROBOTDEV=${IMAGE_ORG_DATAROBOTDEV}/${IMAGE_REPO}:${IMAGE_TAG}

echo "GIT_ROOT: $GIT_ROOT"
echo "BUILD_DOCKER_DIR: $BUILD_DOCKER_DIR"

cp -f "$GIT_ROOT/custom_model_runner/requirements.txt" "$BUILD_DOCKER_DIR/requirements_drum.txt"


pushd "$BUILD_DOCKER_DIR" || exit 1
ls -la
echo
echo
echo "Building docker image for DRUM compilation"
docker build -t ${IMAGE_NAME_DATAROBOTDEV} ./
popd || exit 1

# Image is ready at this moment, but:
# * run make on drum to pull in all the java deps;
# * build custom_java_predictor for tests
# * commit the image
docker run -t -v "$GIT_ROOT:/tmp/drum" ${IMAGE_NAME_DATAROBOTDEV} bash -c "cd /tmp/drum/custom_model_runner && make && cd /tmp/drum/tests/functional/custom_java_predictor && mvn package"
docker commit "$(docker ps -lq)" ${IMAGE_NAME_DATAROBOTDEV}
docker tag ${IMAGE_NAME_DATAROBOTDEV} ${IMAGE_NAME_DATAROBOT}

if [ -n "${PUSH}" ] ; then
  docker push ${IMAGE_NAME_DATAROBOT}
  docker push ${IMAGE_NAME_DATAROBOTDEV}
fi

