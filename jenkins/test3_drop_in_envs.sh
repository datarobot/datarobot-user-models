#!/usr/bin/env bash
# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.

set -ex

source "$(dirname "$0")/../tools/image-build-utils.sh"

# Path of the drum wheelfile will be passed
build_drum

DRUM_WHEEL="$(realpath "$(find custom_model_runner/dist/datarobot_drum*.whl)")"
build_all_dropin_env_dockerfiles "$DRUM_WHEEL"

# installing DRUM into the test env is required for push test
pip install -U "$DRUM_WHEEL_REAL_PATH"
# requirements_test may install newer packages for testing, e.g. `datarobot`
pip install -r requirements_test.txt

py.test tests/functional/test_drop_in_environments.py \
        --junit-xml="$CDIR/results_drop_in.xml"
