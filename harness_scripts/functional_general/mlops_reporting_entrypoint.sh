#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. ${script_dir}/../common/common.sh
. ${script_dir}/../../tools/create-and-source-venv.sh

DOCKER_HUB_USERNAME=$1
DOCKER_HUB_SECRET=$2
if [ -n "$HARNESS_BUILD_ID" ]; then
  title "Running within a Harness pipeline."
  [ -z $DOCKER_HUB_SECRET ] && echo "Docker HUB secret is expected as an input argument" && exit 1
  docker login -u $DOCKER_HUB_USERNAME -p $DOCKER_HUB_SECRET || { echo "Docker login failed"; exit 1; }
fi

title "Preparing to test"
apt-get update && apt-get install -y curl

title "Installing requirements for all the tests: requirements_test.txt"
pip install -r requirements_test.txt

pushd custom_model_runner
title "Install drum from source"
pip install .
popd

pytest tests/functional/test_mlops_monitoring.py
