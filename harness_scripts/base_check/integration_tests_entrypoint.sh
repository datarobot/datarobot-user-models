#!/usr/bin/env bash

set -exuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. ${script_dir}/../common/create_and_source_venv.sh

pushd custom_model_runner
make
popd  # custom_model_runner

pip3 install -U pip
pip3 install -r requirements_test_unit.txt
pip3 install -e custom_model_runner/
pytest -v tests/integration --junit-xml=results.tests.xml