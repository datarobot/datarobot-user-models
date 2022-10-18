#!/usr/bin/env bash
# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
# This file will be executed from the root of the repository in a python3 virtualenv.
# It will run the test of drum inside a predefined docker image:

DOCKER_IMAGE="datarobot/drum_integration_tests_base"
CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"
FULL_PATH_CODE_DIR=$(realpath $CDIR)

# Installing and configuring java/javac 11 in the jenkins worker
sudo apt update
sudo apt install --no-install-recommends -y openjdk-11-jdk openjdk-11-jre zip
sudo update-alternatives --set java /usr/lib/jvm/java-11-openjdk-amd64/bin/java
sudo update-alternatives --auto javac
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

echo "DEBUG: print java version"
java -version
javac -version


echo "FULL_PATH_CODE_DIR: $FULL_PATH_CODE_DIR"

echo "Running tests inside docker:"
cd $FULL_PATH_CODE_DIR || exit 1


echo
echo "--- env ----"
export
echo
echo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

START_TIME=$(date +%s)

GIT_ROOT=$(git rev-parse --show-toplevel)
echo "GIT_ROOT: $GIT_ROOT"

pip install -U pip


# The "jenkins_artifacts" folder is created in the groovy script
DRUM_WHEEL_REAL_PATH="$(realpath "$(find jenkins_artifacts/datarobot_drum*.whl)")"
echo "DRUM_WHEEL_REAL_PATH: $DRUM_WHEEL_REAL_PATH"
echo
echo "--> Installing wheel"
echo
pip install "${DRUM_WHEEL_REAL_PATH}"


# Change every environment Dockerfile to install freshly built DRUM wheel
source "${GIT_ROOT}/tools/image-build-utils.sh"
echo "--> Change every environment Dockerfile to install local freshly built DRUM wheel: ${DRUM_WHEEL_REAL_PATH}"
build_all_dropin_env_dockerfiles "${DRUM_WHEEL_REAL_PATH}"

source $GIT_ROOT/tests/drum/integration-helpers.sh


CMRUNNER_REQUIREMENT_PATH=$GIT_ROOT/custom_model_runner/requirements.txt

# shellcheck disable=SC2218
build_docker_image_with_cmrun tests/fixtures/cmrun_docker_env \
                               cmrunner_test_env_python_sklearn \
                               $DRUM_WHEEL_REAL_PATH \
                               $CMRUNNER_REQUIREMENT_PATH || exit 1

echo
echo "Installing the requirements for all tests:  $GIT_ROOT/requirements_dev.txt"

pip install -r $GIT_ROOT/requirements_dev.txt -i https://artifactory.int.datarobot.com/artifactory/api/pypi/python-all/simple

echo
echo "Running pytest:"

cd $GIT_ROOT || exit 1
DONE_PREP_TIME=$(date +%s)

# > NOTE: when pinning datarobot-mlops to 8.2.1 and higher you may need to reinstall datarobot package
# as datarobot-mlops overwrites site-packages/datarobot. [AGENT-3504]
pip install datarobot-mlops==8.2.7


pytest tests/drum/ \
       -k "not test_inference_custom_java_predictor.py and not test_mlops_monitoring.py" \
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


echo "Done running tests: $TEST_RESULT_2"
exit $TEST_RESULT_2
