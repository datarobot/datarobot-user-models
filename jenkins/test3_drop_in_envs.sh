#!/usr/bin/env bash

CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"

pip install -r requirements_test.txt
py.test tests/functional/test_drop_in_environments.py \
        tests/functional/test_inference_model_templates.py \
        tests/functional/test_training_model_templates.py \
        tests/functional/test_dev_envs.py \
        --junit-xml="$CDIR/results_drop_in.xml"
