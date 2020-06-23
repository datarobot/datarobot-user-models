#!/usr/bin/env bash

echo
echo "--- env ----"
export
echo
echo

# We assume for now that this scripts starting point is the top dir
CODE_DIR=$(pwd)

START_TIME=$(date +%s)
cd /opt || exit 1
. v3.7/bin/activate

echo "-- running drum tests - assuming running inside Docker"
cd  $CODE_DIR || exit 1
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

function build_docker_image_with_cmrun() {
  cd "$GIT_ROOT" || exit 1

  orig_docker_context_dir=$1
  image_name=$2
  drum_wheel=$3
  drum_requirements=$4

  docker_dir=/tmp/cmrun_docker.$$

  echo "Building docker image:"
  echo "orig_docker_context_dir: $orig_docker_context_dir"
  echo "image_name:              $image_name"
  echo "deum_wheel:              $drum_wheel"
  echo "drum_requirements:       $drum_requirements"
  echo "docker_dir:              $docker_dir"

  rm -rf $docker_dir
  cp -a $orig_docker_context_dir $docker_dir

  cp $drum_wheel $docker_dir
  cp $drum_requirements $docker_dir/drum_requirements.txt

  cd $docker_dir || exit 1
  docker build -t $image_name ./
  rm -rf $docker_dir
  echo
  echo
  docker images
  cd $GIT_ROOT || exit 1
}

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
#  --junit-xml="$CODE_DIR/results_integration.xml"
pytest tests/drum/test_units.py tests/drum/test_custom_model.py --junit-xml="$CODE_DIR/results_integration.xml"

TEST_RESULT=$?
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
PREP_TIME=$((DONE_PREP_TIME - START_TIME))
TEST_TIME=$((END_TIME - DONE_PREP_TIME))

echo "Total test time: $TOTAL_TIME"
echo "Prep time:     : $PREP_TIME"
echo "Test time:     : $TEST_TIME"

exit $TEST_RESULT
