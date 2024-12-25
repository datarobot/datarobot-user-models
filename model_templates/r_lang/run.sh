#!/usr/bin/env bash

script_name=$(basename ${BASH_SOURCE[0]})
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# working_dir=${script_dir}/../..
working_dir=/opt/code

pushd ${working_dir}

# PYTHONPATH=./custom_model_runner DEBUG=1 ./custom_model_runner/bin/drum \
drum \
    score \
    --code-dir ./model_templates/r_lang \
    --target-type regression \
    --input tests/testdata/juniors_3_year_stats_regression.csv

popd
