#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. ${script_dir}/../common/common.sh
. ${script_dir}/../../tools/create-and-source-venv.sh

title "Preparing to test"
apt-get update && apt-get install -y curl

title "Installing requirements for all the tests: requirements_test.txt"
pip install -r requirements_test.txt

pushd custom_model_runner
title "Install drum from source"
pip install .
popd

pytest tests/functional/test_mlops_monitoring.py
