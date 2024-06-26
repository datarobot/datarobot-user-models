#!/usr/bin/env bash
script_name=$(basename ${BASH_SOURCE[0]})
script_dir=$(realpath $(dirname ${BASH_SOURCE[0]}))

mlpiper_py_root="$script_dir/../.."
resources_root="$mlpiper_py_root/tests/resources"
model_filepath="$resources_root/models/sklearn_kmeans_rest_model.pkl"
pipeline_filepath="$resources_root/pipelines/echo-rest-model-serving-pipeline.json"
components_root="$resources_root/steps/RestModelServing/restful_dummy_echo_test"
deployment_path=$(mktemp -d -t mlpiper-deployment-XXXXXXXXXX)

set -x
PYTHONPATH=$mlpiper_py_root $mlpiper_py_root/bin/mlpiper run \
    --input-model $model_filepath \
    -f $pipeline_filepath \
    -r $components_root \
    -d $deployment_path \
    --force
set +x

# rm -rf $deployment_path

######################################################################################################
# Use the following command to send predictions:
#   curl -X POST -H "Content-Type: application/json" http://localhost:8888/predict -d '{"data": [1, 2]}'
# or,
#   while true; do curl -X POST -H "Content-Type: application/json" http://localhost:8888/predict -d '{"data": [1, 2]}'; done