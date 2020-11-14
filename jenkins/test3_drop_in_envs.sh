#!/usr/bin/env bash

CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"

# installing DRUM into the test env is required for push test
#pip install datarobot-drum
# requirements_test may require newer packages for testing, e.g. `datarobot`
pip install -r requirements_test.txt

py.test tests/functional/test_drop_in_environments.py \
        tests/functional/test_inference_model_templates.py \
        tests/functional/test_training_model_templates.py \
        --junit-xml="$CDIR/results_drop_in.xml"
