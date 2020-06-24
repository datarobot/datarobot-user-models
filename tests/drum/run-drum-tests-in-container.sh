#!/usr/bin/env bash

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

cd $GIT_ROOT/custom_model_runner || exit 1

echo
echo "--> Building wheel"
echo
make clean && make
CMRUNNER_WHEEL=$(find dist/datarobot_drum*.whl)
CMRUNNER_WHEEL_REAL_PATH=$(realpath $CMRUNNER_WHEEL)

echo "CMRUNNER_WHEEL_REAL_PATH: $CMRUNNER_WHEEL_REAL_PATH"

echo
echo "--> Installing wheel"
echo
pip install "${CMRUNNER_WHEEL}[R]"

source $GIT_ROOT/tests/drum/integration-helpers.sh

cd $GIT_ROOT || exit 1

CMRUNNER_REQUIREMENT_PATH=$GIT_ROOT/custom_model_runner/requirements.txt

# shellcheck disable=SC2218
build_docker_image_with_cmrun tests/fixtures/cmrun_docker_env \
                               cmrunner_test_env_python_sklearn \
                               $CMRUNNER_WHEEL_REAL_PATH \
                               $CMRUNNER_REQUIREMENT_PATH || exit 1

echo
echo "Installing the requirements for all tests:  $GIT_ROOT/requirements.txt"
cd $GIT_ROOT || exit 1
pip install -r $GIT_ROOT/requirements.txt

echo
echo "Running pytest:"

cd $GIT_ROOT || exit 1
DONE_PREP_TIME=$(date +%s)

#pytest -s tests/drum/test_custom_model.py::TestCMRunner::test_custom_models_with_drum[rds-regression-R-None] \
#  --junit-xml="$GIT_ROOT/results_integration.xml"
pytest tests/drum/test_units.py tests/drum/test_custom_model.py --junit-xml="$GIT_ROOT/results_integration.xml"

TEST_RESULT=$?
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
PREP_TIME=$((DONE_PREP_TIME - START_TIME))
TEST_TIME=$((END_TIME - DONE_PREP_TIME))

echo "Total test time: $TOTAL_TIME"
echo "Prep time:     : $PREP_TIME"
echo "Test time:     : $TEST_TIME"

exit $TEST_RESULT
