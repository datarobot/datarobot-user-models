#!/usr/bin/env bash
# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.

echo
echo "--- env ----"
export
echo
echo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

START_TIME=$(date +%s)
cd /opt || exit 1
. v3.7/bin/activate

echo "-- running drum tests - assuming running inside Docker"
cd $SCRIPT_DIR || exit 1
pwd
GIT_ROOT=$(git rev-parse --show-toplevel)
echo "GIT_ROOT: $GIT_ROOT"

if [ -n "${PIPELINE_CONTROLLER}" ]; then
  # The "jenkins_artifacts" folder is created in the groovy script
  DRUM_WHEEL_REAL_PATH="$(realpath "$(find jenkins_artifacts/datarobot_drum*.whl)")"
else
  pushd $GIT_ROOT/custom_model_runner || exit 1
  echo
  echo "--> Building wheel"
  echo
  make clean && make
  DRUM_WHEEL_REAL_PATH="$(realpath "$(find dist/datarobot_drum*.whl)")"
  pop
fi

echo "DRUM_WHEEL_REAL_PATH: $DRUM_WHEEL_REAL_PATH"

echo
echo "--> Installing wheel"
echo
pip install "${DRUM_WHEEL_REAL_PATH}[R]"
popd


# Change every environment Dockerfile to install freshly built DRUM wheel
source "${GIT_ROOT}/tools/image-build-utils.sh"
echo "--> Change every environment Dockerfile to install local freshly built DRUM wheel: ${DRUM_WHEEL_REAL_PATH}"
build_all_dropin_env_dockerfiles "${DRUM_WHEEL_REAL_PATH}"

source $GIT_ROOT/tests/drum/integration-helpers.sh

cd $GIT_ROOT || exit 1

CMRUNNER_REQUIREMENT_PATH=$GIT_ROOT/custom_model_runner/requirements.txt

# shellcheck disable=SC2218
build_docker_image_with_cmrun tests/fixtures/cmrun_docker_env \
                               cmrunner_test_env_python_sklearn \
                               $DRUM_WHEEL_REAL_PATH \
                               $CMRUNNER_REQUIREMENT_PATH || exit 1

echo
echo "Installing the requirements for all tests:  $GIT_ROOT/requirements_dev.txt"
cd $GIT_ROOT || exit 1
pip install -r $GIT_ROOT/requirements_dev.txt -i https://artifactory.int.datarobot.com/artifactory/api/pypi/python-all/simple

echo
echo "Running pytest:"

cd $GIT_ROOT || exit 1
DONE_PREP_TIME=$(date +%s)

pip install \
    --extra-index-url https://artifactory.int.datarobot.com/artifactory/api/pypi/python-all/simple \
    datarobot-mlops

pytest tests/drum/ \
       -m "not sequential" \
       --junit-xml="$GIT_ROOT/results_integration.xml" \
       -n auto

TEST_RESULT_2=$?

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
PREP_TIME=$((DONE_PREP_TIME - START_TIME))
TEST_TIME=$((END_TIME - DONE_PREP_TIME))

echo "Total test time: $TOTAL_TIME"
echo "Prep time:     : $PREP_TIME"
echo "Test time:     : $TEST_TIME"

if [ $TEST_RESULT_2 -ne 0 ] ; then
  echo "Got error in one of the tests:"
  echo "Non sequential tests: $TEST_RESULT_2"
  exit 1
else
  exit 0
fi
