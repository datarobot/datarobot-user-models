#!/usr/bin/env bash
# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.

# This is a shortcut script to only build the integration test docker env

GIT_ROOT=$(git rev-parse --show-toplevel)
source $GIT_ROOT/tests/functional/integration-helpers.sh

cd $GIT_ROOT || exit 1

CMRUNNER_REQUIREMENT_PATH=$GIT_ROOT/custom_model_runner/requirements.txt
DRUM_WHEEL_REAL_PATH=$(find custom_model_runner/dist/datarobot_drum**.whl)

# shellcheck disable=SC2218
build_docker_image_with_cmrun tests/fixtures/cmrun_docker_env \
                               cmrunner_test_env_python_sklearn \
                               $DRUM_WHEEL_REAL_PATH \
                               $CMRUNNER_REQUIREMENT_PATH || exit 1