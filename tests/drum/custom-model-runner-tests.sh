#!/usr/bin/env bash

# This script is desined to perform an integration tests for running
# custom models with 'mliper' on Py3 environment.
# Py2 environvment requires debugging because most of the deps fail to compile.
# This script is supposed to run on a local development environment and assumes
# that virtualenvwrapper package is intalled there.
#
# Here are the steps that are executed:
#   1. Load 'virtualenvwrapper' environment.
#   2. If virtual environment doesn't exist or --reinstall flag is provided,
#      create the virtual environment for the given Python interperter.
#   3. Install provided 'mlpiper' wheel.
#   4. Install the requirements for the integration tests.
#   5. Run the integration tests.

REINSTALL=0
if [ "$1" = "--reinstall" ]; then
  REINSTALL=1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd ${SCRIPT_DIR}
CUSTOM_TEMPLATES_ROOT=$(git rev-parse --show-toplevel)
# CUSTOM_TEMPLATES_TESTS_PATH=${CUSTOM_TEMPLATES_ROOT}/tests
CUSTOM_MODEL_RUNNER_DIR=${CUSTOM_TEMPLATES_ROOT}/custom_model_runner
VENV_BASE_NAME="custom-model-runner-test"

. /usr/local/bin/virtualenvwrapper.sh

source ${SCRIPT_DIR}/integration-helpers.sh

function build_docker_image_with_cmrun() {
  orig_docker_context_dir=$1
  image_name=$2

  docker_dir=/tmp/cmrun_docker.$$

  rm -rf $docker_dir
  cp -a $orig_docker_context_dir $docker_dir

  pushd ${CUSTOM_MODEL_RUNNER_DIR}
  make clean && make
  CMRUNNER_WHEEL=$(find dist/datarobot_drum*.whl)
  cp $CMRUNNER_WHEEL $docker_dir
  cp requirements.txt $docker_dir/drum_requirements.txt
  popd

  pushd $docker_dir
  docker build -t $image_name ./
  popd
  rm -rf $docker_dir
  echo

}

function install_and_test() {
    py_ver=$1
    venv="${VENV_BASE_NAME}-py${py_ver}"

    venv_exists_check ${venv}
    RESULT=$?

    if [ $RESULT -ne 0 ] || [ ${REINSTALL} -eq 1 ]; then
        create_virtual_env ${venv} ${py_ver}
    fi

    workon ${venv}

    # 5. Test custom model runner
    title "Testing Custom Model Runner on '${venv}' ..."
    pip install pytest -r ${CUSTOM_TEMPLATES_ROOT}/requirements.txt

    pushd ${CUSTOM_MODEL_RUNNER_DIR}

    pip uninstall -y datarobot-drum

    make clean && make
    CMRUNNER_WHEEL=$(find dist/datarobot_drum*.whl)
    pip install "${CMRUNNER_WHEEL}[R]"
    pip install --no-deps -U dist/datarobot_drum*.whl

    popd

    #pytest -s ${SCRIPT_DIR}/test_custom_model.py::TestCMRunner::test_custom_models_with_cmrunner_prediction_server_docker
    #pytest -s ${SCRIPT_DIR}/test_custom_model.py::TestCMRunner::test_custom_models_with_cmrunner[rds-regression-R]
    pytest -s ${SCRIPT_DIR}/test_custom_model.py
    deactivate
}


build_docker_image_with_cmrun $CUSTOM_TEMPLATES_ROOT/tests/fixtures/cmrun_docker_env \
                              cmrunner_test_env_python_sklearn

# Install and test mlpiper on Python3
install_and_test 3



echo
title "custom model runner tests passed successfuly on Python3 environment"
