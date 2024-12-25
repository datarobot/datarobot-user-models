#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. ${script_dir}/../common/create_and_source_venv.sh

pushd ./tests/functional/custom_java_predictor
mvn package
popd

pushd custom_model_runner
echo "== Build java entrypoint, base predictor and install DRUM from source =="
make java_components
pip install .
popd

pytest tests/functional/test_inference_custom_java_predictor.py