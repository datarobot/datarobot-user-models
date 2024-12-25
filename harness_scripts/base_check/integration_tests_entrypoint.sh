#!/usr/bin/env bash

set -exuo pipefail
cd custom_model_runner
make
cd -
python3 -m venv /tmp/venv
. /tmp/venv/bin/activate
pip3 install -U pip
pip3 install -r requirements_test_unit.txt
pip3 install -e custom_model_runner/
pytest -v tests/integration --junit-xml=results.tests.xml