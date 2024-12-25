#!/usr/bin/env bash

script_name=$(basename ${BASH_SOURCE[0]})
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

working_dir=${script_dir}/../..

export DRUM_JAVA_CUSTOM_PREDICTOR_CLASS=custom.CustomModel
export DRUM_JAVA_CUSTOM_CLASS_PATH=${script_dir}/model/custom-model-0.1.0.jar

pushd ${working_dir}

PYTHONPATH=./custom_model_runner DEBUG=1 ./custom_model_runner/bin/drum \
    score \
    --code-dir ./model_templates/java_unstructured/model \
    --target-type unstructured \
    --input ./tests/testdata/unstructured_data.txt

popd
