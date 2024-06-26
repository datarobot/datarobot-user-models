#!/usr/bin/env bash

# This script is designed to perform a sanity-check for 'mlpiper' tool on a cleanup
# environments for both Python2 and Python3. It executes a generic pipeline, which
# drives a sklearn steps. This script is supposed to run on a local development
# environment and assumes that virtualenvwrapper package is intalled there.
#
# Here are the steps that are executed:
#   1. Load 'virtualenvwrapper' environment
#   2. Create 'mlpiper' wheels on the existing development environment
#   3. The following steps are performed separately for Python2 and Python3:
#       a. If exists, remove the virtual environment that is used for testing
#       b. Create the virtual environment for the given Python interperter: 2 and 3 separately
#       c. Install the relevant 'mlpiper' wheel for the relevant Python interperter
#       d. Install the requirements for the sanity test example
#       e. Run the sanity test


SCRIPT_NAME=$(basename ${BASH_SOURCE[0]})
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

USAGE="Usage: ${SCRIPT_NAME} [--local] [--help]"
local=0
while [ "$1" != "" ]; do
    case $1 in
        --local)        shift; local=1 ;;
        --help)         echo $USAGE; exit 0 ;;
    esac
    shift
done

MLPIPER_ROOT=$(git rev-parse --show-toplevel)
MLPIPER_PY_ROOT=${MLPIPER_ROOT}/mlpiper-py
MLPIPER_TEST_RESOURCES_PATH="${MLPIPER_PY_ROOT}/tests/resources"
MLPIPER_INTEG_SANITY_PIPELINE=${MLPIPER_TEST_RESOURCES_PATH}/pipelines/simple_sklearn_pipeline.json
MLPIPER_INTEG_SANITY_STEPS_ROOT=${MLPIPER_TEST_RESOURCES_PATH}/steps/Generic/Python/sklearn
MLPIPER_INTEG_SANITY_DATASET=${MLPIPER_TEST_RESOURCES_PATH}/datasets/10k_diabetes.csv

INTEGRATION_SANITY_TEST_PATH=${MLPIPER_PY_ROOT}/mlpiper/integration/sklearn
VENV_BASE_NAME="mlpiper-sanity-test"

. /usr/local/bin/virtualenvwrapper.sh

. ${SCRIPT_DIR}/integration-helpers.sh
xargs_tool=$(setup_xargs_tool)

title "Preparing 'mlpiper' distribution ..."
cd $MLPIPER_ROOT
make dist || { title "Failed building 'mlpiper' distribution! Exiting ..."; exit -1; }

function install_deps() {
  title "Installing sanity dependencies ..."

  tmp_requirements_filepath=$(mktemp '/tmp/mlpiper-sanity-requirements.txt.XXXXX')
  rm -rf ${tmp_requirements_filepath}

  mlpiper deps \
      -f ${MLPIPER_INTEG_SANITY_PIPELINE} \
      --comp-root ${MLPIPER_INTEG_SANITY_STEPS_ROOT} \
      --output-path ${tmp_requirements_filepath} \
      Python \
      || { title "Error running test! Exiting ..."; exit -1; }

  cat ${tmp_requirements_filepath} | $xargs_tool -n 1 -L 1 -d '\n' pip install

  rm ${tmp_requirements_filepath}
}

function install_and_test() {
    py_ver=$1
    venv="${VENV_BASE_NAME}-py${py_ver}"

    if [[ ${local} == 0 ]]; then
        create_mlpiper_virtual_env ${venv} ${py_ver} $(find ${MLPIPER_PY_ROOT}/dist/mlpiper*-py2.py3-*.whl)
        workon ${venv}
    fi

    install_deps

    title "Test mlpiper on '${venv}' ..."

    time INTEG_CSV_DATASET=${MLPIPER_INTEG_SANITY_DATASET} mlpiper --skip-mlpiper-deps \
        run \
        -f ${MLPIPER_INTEG_SANITY_PIPELINE} \
        --comp-root ${MLPIPER_INTEG_SANITY_STEPS_ROOT} \
        || { title "Error running test! Exiting ..."; exit -1; }

    [[ ${local} == 0 ]] && deactivate
}

# Install and test mlpiper on Python3
install_and_test 3

# Install and test mlpiper on Python2
install_and_test 2

echo
title "'mlpiper' sanity test passed successfully on both Python2 & Python3"
