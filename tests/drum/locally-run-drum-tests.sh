#!/usr/bin/env bash

# This script creates a python environment and runs drum integration tests locally.
# It installs only python dependencies. Also R and JDK 11 are required.
#
# Here are the steps that are executed:
#   1. Script checks if virtual env `drum_tests_virtual_environment` exists in the HOME dir.
#   2. If virtual environment doesn't exist it is created.
#   3. drum wheel is compiled and installed.
#   4. As part of test cases a docker image `cmrunner_test_env_python_sklearn` for sklearn environment is created.
#   5. Run the integration tests.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd ${SCRIPT_DIR}
GIT_ROOT=$(git rev-parse --show-toplevel)
cd ${GIT_ROOT}
CUSTOM_MODEL_RUNNER_DIR=${GIT_ROOT}/custom_model_runner
PATH_TO_ENV="${HOME}/drum_tests_virtual_environment"
ENV_ACTIVATE=${PATH_TO_ENV}/bin/activate


source ${SCRIPT_DIR}/integration-helpers.sh


function check_env_or_create() {
    echo "Looking for virtual env in: ${PATH_TO_ENV}"
    if [ ! -d ${PATH_TO_ENV} ]; then
        echo "Creating virtual environment"
        python3 -m pip show virtualenv
        if [ $? -eq 1 ]
        then
          python3 -m pip install virtualenv
        fi
        python3 -m virtualenv ${PATH_TO_ENV}
    fi

    echo "Activating virtual env in: ${PATH_TO_ENV}"
    if [ ! -f $ENV_ACTIVATE ]; then
        echo "'$ENV_ACTIVATE' not found"
        echo "'$PATH_TO_ENV' expected to be virtual environment"
        exit 1
    fi
    . $ENV_ACTIVATE
    pip install pytest -r ${GIT_ROOT}/requirements_dev.txt
    deactivate
}

function test_drum() {
    CMRUNNER_WHEEL=$1

    . $ENV_ACTIVATE

    # 5. Test custom model runner
    title "Testing Custom Model Runner in '${PATH_TO_ENV}' ..."

    pushd ${CUSTOM_MODEL_RUNNER_DIR}

    pip uninstall -y datarobot-drum

    pip install "${CMRUNNER_WHEEL}[R]"
    pip install --no-deps -U ${CMRUNNER_WHEEL}

    popd

    #pytest -s ${SCRIPT_DIR}/test_custom_model.py::TestCMRunner::test_custom_models_with_cmrunner_prediction_server_docker
    #pytest -s ${SCRIPT_DIR}/test_custom_model.py::TestCMRunner::test_custom_models_with_cmrunner[rds-regression-R]

    # ignore local failures for drum push and mlops
    pytest -s ${SCRIPT_DIR}/test_*
    deactivate
}
#######################################################################################################################
#######################################################################################################################

check_env_or_create

# build drum wheel
. $ENV_ACTIVATE
pushd ${CUSTOM_MODEL_RUNNER_DIR}
make
CMRUNNER_WHEEL=$(find dist/datarobot_drum*.whl)
CMRUNNER_WHEEL_REAL_PATH=$(realpath $CMRUNNER_WHEEL)
CMRUNNER_REQUIREMENT_PATH=$GIT_ROOT/custom_model_runner/requirements.txt
popd

# build container for sklearn environment
build_docker_image_with_cmrun $GIT_ROOT/tests/fixtures/cmrun_docker_env \
                              cmrunner_test_env_python_sklearn \
                              $CMRUNNER_WHEEL_REAL_PATH \
                              $CMRUNNER_REQUIREMENT_PATH || exit 1
deactivate

# Test drum on Python3
test_drum ${CMRUNNER_WHEEL_REAL_PATH}
title "drum tests passed successfully on Python3 environment"
