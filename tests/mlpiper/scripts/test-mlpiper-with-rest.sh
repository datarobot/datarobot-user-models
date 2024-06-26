#!/usr/bin/env bash
script_name=$(basename ${BASH_SOURCE[0]})
script_dir=$(realpath $(dirname ${BASH_SOURCE[0]}))

mlpiper_py_root="$script_dir/../.."
resources_root="$mlpiper_py_root/tests/resources"
model_filepath="$resources_root/models/sklearn_kmeans_rest_model.pkl"
pipeline_filepath="$resources_root/pipelines/sklearn-rest-model-serving-pipeline.json"
components_root="$resources_root/steps/RestModelServing/restful_sklearn_serving_test"
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
