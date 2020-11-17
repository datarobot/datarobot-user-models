#!/usr/bin/env bash

CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"
DRUM_BUILDER_IMAGE="datarobot/drum-builder"
echo $CDIR

# pull DRUM builder container and build DRUM wheel
docker pull ${DRUM_BUILDER_IMAGE}

# If we are in terminal will be true when running the script manually. Via Jenkins it will be false
TERMINAL_OPTION=""
if [ -t 1 ] ; then
  TERMINAL_OPTION="-t"
fi

docker run -i ${TERMINAL_OPTION} --user $(id -u):$(id -g) -v $CDIR:/tmp/drum $DRUM_BUILDER_IMAGE bash -c "cd /tmp/drum/custom_model_runner && make"
docker rmi $DRUM_BUILDER_IMAGE --force

DRUM_WHEEL=$(find custom_model_runner/dist/datarobot_drum*.whl)
DRUM_WHEEL_FILENAME=$(basename $DRUM_WHEEL)
DRUM_WHEEL_REAL_PATH=$(realpath $DRUM_WHEEL)

# Change every environment Dockerfile to install fresly build DRUM wheel
pushd public_dropin_environments
DIRS=$(ls)
for d in $DIRS
do
  pushd $d
  cp $DRUM_WHEEL_REAL_PATH .
  echo "
COPY $DRUM_WHEEL_FILENAME $DRUM_WHEEL_FILENAME
RUN pip3 install -U $DRUM_WHEEL_FILENAME && rm -rf $DRUM_WHEEL_FILENAME" >> Dockerfile
  popd
done
popd

# installing DRUM into the test env is required for push test
pip install -U $DRUM_WHEEL_REAL_PATH
# requirements_test may install newer packages for testing, e.g. `datarobot`
pip install -r requirements_test.txt

# put tests in this exact order as they build images and as a result jenkins instance may run out of space
py.test tests/functional/test_inference_model_templates.py \
        tests/functional/test_training_model_templates.py \
        tests/functional/test_drop_in_environments.py \
        tests/functional/test_drum_push.py \
        --junit-xml="$CDIR/results_drop_in.xml"
