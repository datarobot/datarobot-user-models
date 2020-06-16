#!/usr/bin/env bash

CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"

py.test tests/functional/test_custom_training_models.py \
        --junit-xml="$CDIR/results_drop_in.xml"
