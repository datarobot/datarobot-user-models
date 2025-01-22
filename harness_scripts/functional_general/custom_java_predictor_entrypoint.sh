#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. ${script_dir}/../../tools/create-and-source-venv.sh

pushd ./tests/functional/custom_java_predictor
mvn package
popd

pushd custom_model_runner
echo "== Build java entrypoint, base predictor and install DRUM from source =="
make java_components
pip install .
popd

echo "Installing requirements for all the tests: requirements_test.txt"
pip install -r requirements_test.txt

echo "Installing datarobot_drum from source code"
pip3 install -e custom_model_runner/

pytest tests/functional/test_inference_custom_java_predictor.py
