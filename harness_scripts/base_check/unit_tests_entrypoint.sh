#!/usr/bin/env bash

set -exuo pipefail
pip install -U pip
pip install -r requirements_test_unit.txt
pip install -e custom_model_runner/
pytest -v tests/unit --junit-xml=results.tests.xml