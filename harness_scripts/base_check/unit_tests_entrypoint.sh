#!/usr/bin/env bash

set -exuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. ${script_dir}/../common/create_and_source_venv.sh

pip install -r requirements_test_unit.txt
pip install -e custom_model_runner/
pytest -v tests/unit --junit-xml=results.tests.xml