#!/usr/bin/env bash
# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.

set -ex
GIT_ROOT=$(git rev-parse --show-toplevel)

source "$(dirname "$0")/../tools/image-build-utils.sh"

if [ -n "${PIPELINE_CONTROLLER}" ]; then
  # The "jenkins_artifacts" folder is created in the groovy script
  DRUM_WHEEL_REAL_PATH="$(realpath "$(find jenkins_artifacts/datarobot_drum*.whl)")"
else
  build_drum
  DRUM_WHEEL_REAL_PATH="$(realpath "$(find custom_model_runner/dist/datarobot_drum*.whl)")"
fi

build_all_dropin_env_dockerfiles "$DRUM_WHEEL_REAL_PATH"

pip install pip==22.1.2
# installing DRUM into the test env is required for push test
pip install -U $DRUM_WHEEL_REAL_PATH
# requirements_test may install newer packages for testing, e.g. `datarobot`
pip install -r requirements_test.txt

py.test tests/functional/test_inference_model_templates.py \
        --junit-xml="${GIT_ROOT}/results_drop_in.xml"

