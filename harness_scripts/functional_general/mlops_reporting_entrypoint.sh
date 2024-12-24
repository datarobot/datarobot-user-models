#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. ${script_dir}/../common/create_and_source_venv.sh

echo "== Preparing to test =="
apt-get update && apt-get install -y curl

echo "Installing requirements for all the tests: requirements_test.txt"
pip install -r requirements_test.txt

pushd custom_model_runner
echo "== Install drum from source =="
pip install .
popd

pytest tests/functional/test_mlops_monitoring.py