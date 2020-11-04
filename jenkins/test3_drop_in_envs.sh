#!/usr/bin/env bash

CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"

pip install -r requirements_test.txt
# installing DRUM into the test env is required for push test
pip install datarobot-drum
py.test tests/functional/test_drum_push.py \
        --junit-xml="$CDIR/results_drop_in.xml"
