#!/usr/bin/env bash

set -exuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. ${script_dir}/../../tools/create-and-source-venv.sh

pushd custom_model_runner
make
popd  # custom_model_runner

pip3 install -r requirements_test_unit.txt
pip3 install -e custom_model_runner/ --verbose
pytest -v tests/integration --junit-xml=results.tests.xml