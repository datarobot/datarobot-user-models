#!/usr/bin/env bash

script_name=$(basename ${BASH_SOURCE[0]})
script_dir=$(realpath $(dirname ${BASH_SOURCE[0]}))

USAGE="Usage: ${script_name} [--help] [--dir <artifacts-dir>] [--skip-train] [--skip-predict] [--log-level <level>]"

function validate_arg {
    [ -z $2 ] && { echo "Missing '$1' arg value!"; exit -1; }
}

artifacts_dir=""
skip_train=0
skip_predict=0
log_level="info"
while [ -n "$1" ]; do
    case $1 in
        --dir)          artifacts_dir=$2; validate_arg $1 $2 ; shift ;;
        --skip-train)   skip_train=1 ;;
        --skip-predict) skip_predict=1 ;;
        --log-level)    log_level=$2; validate_arg $1 $2 ; shift ;;
        --help)         echo $USAGE; exit 0 ;;
        *)              echo $USAGE; exit 0 ;;
    esac
    shift
done

# General
artifacts_dir_prefix="/tmp/sagemaker-mnist-artifacts-"
if [ -z ${artifacts_dir} ]; then
    artifacts_dir=$(mktemp -d ${artifacts_dir_prefix}XXXXX)
else
    if [[ ${skip_train} == 0 ]]; then
        rm -rf ${artifacts_dir}
        mkdir -p ${artifacts_dir}
    fi
fi
echo "Artifacts dir: ${artifacts_dir}"

mlpiper_py_root="$script_dir/../.."
resources_root="$mlpiper_py_root/tests/resources"
components_root="$resources_root/steps/SageMaker"
pipelines_root="$resources_root/pipelines"
model_filepath="${artifacts_dir}/sagemaker-kmeans-model.tar.gz"

function pipeline_with_creds {
    pipeline_name=$1
    pipeline_filepath=${pipelines_root}/${pipeline_name}
    local pipeline_with_creds_filepath="/tmp/${pipeline_name}"

    cp ${pipeline_filepath} ${pipeline_with_creds_filepath}

    region=$(grep region ~/.aws/config | awk '{print $3}')
    aws_access_key_id=$(grep aws_access_key_id ~/.aws/credentials | awk '{print $3}')
    aws_secret_access_key=$(grep aws_secret_access_key ~/.aws/credentials | awk '{print $3}')

    sed -i'' -e "s/__REGION_PLACEHOLDER__/${region}/g" ${pipeline_with_creds_filepath}
    sed -i'' -e "s/__AWS_ACCESS_KEY_ID_PLACEHOLDER__/${aws_access_key_id}/g" ${pipeline_with_creds_filepath}
    sed -i'' -e "s/__AWS_SECRET_ACCESS_KEY_PLACEHOLDER__/${aws_secret_access_key}/g" /${pipeline_with_creds_filepath}

    echo ${pipeline_with_creds_filepath}
}

if [[ ${skip_train} == 0 ]]; then
    echo
    echo ################################################################################
    echo # Training
    echo #
    deployment_path=${artifacts_dir}/train-mlpiper-deployment
    tmp_pipeline_filepath=$(pipeline_with_creds "sagemaker_mnist_training.json")

    set -x
    PYTHONPATH=${mlpiper_py_root} $mlpiper_py_root/bin/mlpiper --logging-level ${log_level} \
        run \
        --output-model ${model_filepath} \
        -f ${tmp_pipeline_filepath} \
        -r ${components_root} \
        -d ${deployment_path} \
        --force
    set +x

    rm -rf ${tmp_pipeline_filepath}

    if [ -f $model_filepath ]; then
        rm -rf $deployment_path
        echo
        echo "Training passed successfully! Model downloaded to: $model_filepath"
    else
        echo "Training failed!"
        exit -1
    fi
fi

if [[ ${skip_predict} == 0 ]]; then
    echo
    echo ################################################################################
    echo # Prediction
    echo #

    pipeline_comp_attr_download_results=$(cat $script_dir/../pipelines/sagemaker_mnist_prediction.json |\
        python3 -c "import json; import sys; print(json.load(sys.stdin)['pipe'][0]['arguments']['local_filepath'])")

    deployment_path=${artifacts_dir}/predict-mlpiper-deployment
    tmp_pipeline_filepath=$(pipeline_with_creds "sagemaker_mnist_prediction.json")
    prediction_results_filepath="${artifacts_dir}/sagemaker_mnist_test_dataset.out"

    set -x
    PYTHONPATH=${mlpiper_py_root} ${mlpiper_py_root}/bin/mlpiper  --logging-level ${log_level} \
        run \
        --input-model ${model_filepath} \
        -f ${tmp_pipeline_filepath} \
        -r ${components_root} \
        -d ${deployment_path} \
        --force
    set +x

    rm -rf ${tmp_pipeline_filepath}

    if [ -f ${pipeline_comp_attr_download_results} ]; then
        cp ${pipeline_comp_attr_download_results} ${prediction_results_filepath}
        rm -rf ${deployment_path}
        echo
        echo "Prediction passed successfully! Results are at: ${prediction_results_filepath}"
    else
        echo "Prediction failed!"
        exit -1
    fi
fi

[[ ${artifacts_dir} == "${artifacts_dir_prefix}*" ]] && { rm -rf ${artifacts_dir}; }

echo "Completed"
