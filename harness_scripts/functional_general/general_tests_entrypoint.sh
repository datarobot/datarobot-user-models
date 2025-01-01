#!/usr/bin/env bash

DOCKER_HUB_SECRET=$1
[ -z $DOCKER_HUB_SECRET ] && echo "Docker HUB secret is expected as an input argument" && exit 1
docker login -u datarobotread2 -p $DOCKER_HUB_SECRET

echo "== Build image for tests =="
cp -r custom_model_runner/ public_dropin_environments/python3_sklearn/
cd public_dropin_environments/python3_sklearn/
echo -e "RUN pip uninstall -y datarobot-drum\nCOPY ./custom_model_runner /tmp/custom_model_runner\nRUN pip install /tmp/custom_model_runner" >> Dockerfile

docker build -t python3_sklearn_test_env .
cd -

docker images

echo "== Preparing to test =="
echo "Installing requirements for all the tests: requirements_test.txt"
pip install -r requirements_test.txt

echo "== Uninstall drum =="
pip uninstall -y datarobot-drum

cd custom_model_runner
echo "== Install drum from source =="
pip install .
cd -

pytest tests/functional/test_mlops_monitoring.py::TestMLOpsMonitoring::test_drum_unstructured_model_embedded_monitoring_in_sklearn_env
pytest tests/functional/ -k "not test_inference_custom_java_predictor.py and not test_mlops_monitoring.py" -m "not sequential"
