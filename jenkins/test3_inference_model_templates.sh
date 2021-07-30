#!/usr/bin/env bash

set -ex

source "$(dirname "$0")/../tools/image-build-utils.sh"

build_drum
DRUM_WHEEL="$(realpath "$(find custom_model_runner/dist/datarobot_drum*.whl)")"
build_all_dropin_env_dockerfiles "$DRUM_WHEEL"

# installing DRUM into the test env is required for push test
pip install -U $DRUM_WHEEL_REAL_PATH
# requirements_test may install newer packages for testing, e.g. `datarobot`
pip install -r requirements_test.txt

py.test tests/functional/test_inference_model_templates.py \
        --junit-xml="$CDIR/results_drop_in.xml"

