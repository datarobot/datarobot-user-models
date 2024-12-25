#!/usr/bin/env bash

cd ./tests/functional/custom_java_predictor
mvn package
cd -
cd custom_model_runner
echo "== Build java entrypoint, base predictor and install DRUM from source =="
make java_components
pip install .
cd -
pytest tests/functional/test_inference_custom_java_predictor.py