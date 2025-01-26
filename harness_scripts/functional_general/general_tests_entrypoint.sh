#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. ${script_dir}/../common/common.sh

DOCKER_HUB_SECRET=$1
if [ -n "$HARNESS_BUILD_ID" ]; then
  title "Running within a Harness pipeline."
  [ -z $DOCKER_HUB_SECRET ] && echo "Docker HUB secret is expected as an input argument" && exit 1
  docker login -u datarobotread2 -p $DOCKER_HUB_SECRET || { echo "Docker login failed"; exit 1; }
fi

title "Build image for tests"
tmp_py3_sklearn_env_dir=$(mktemp -d)
cp -r public_dropin_environments/python3_sklearn/* $tmp_py3_sklearn_env_dir
cp -r custom_model_runner/ $tmp_py3_sklearn_env_dir

constants_file="tests/constants.py"
image_name=$(grep -oP '^DOCKER_PYTHON_SKLEARN\s*=\s*"\K[^"]+' "$constants_file")
[ -z $image_name ] && echo "DOCKER_PYTHON_SKLEARN is not set in $constants_file" && exit 1

pushd $tmp_py3_sklearn_env_dir
# remove DRUM from requirements file to be able to install it from source
sed -i "s/^datarobot-drum.*//" requirements.txt
# Update the Dockerfile to install the custom model runner
echo -e "RUN pip uninstall -y datarobot-drum || true\nCOPY ./custom_model_runner /tmp/custom_model_runner\nRUN pip install /tmp/custom_model_runner" >> Dockerfile
docker build -t $image_name .
popd

docker images
echo "Image build succeeded: '$image_name'"

title "Preparing to test"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. ${script_dir}/../../tools/create-and-source-venv.sh

title "Installing requirements for all the tests: requirements_test.txt"
pip install -r requirements_test.txt

pushd custom_model_runner
title "Install drum from source"
pip install .
popd

pytest tests/functional/ -k "not test_inference_custom_java_predictor.py and not test_mlops_monitoring.py" -m "not sequential"
