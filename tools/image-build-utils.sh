#!/usr/bin/env bash

GIT_ROOT=$(git rev-parse --show-toplevel)

function build_drum() {
  CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"
  DRUM_BUILDER_IMAGE="datarobot/drum-builder"

  # pull DRUM builder container and build DRUM wheel
  docker pull ${DRUM_BUILDER_IMAGE}

  # If we are in terminal will be true when running the script manually. Via Jenkins it will be false.
  TERMINAL_OPTION=""
  if [ -t 1 ] ; then
    TERMINAL_OPTION="-t"
  fi

  docker run -i ${TERMINAL_OPTION} --user "$(id -u):$(id -g)" -v $CDIR:/tmp/drum $DRUM_BUILDER_IMAGE bash -c "cd /tmp/drum/custom_model_runner && make"
  docker rmi $DRUM_BUILDER_IMAGE --force

}

function build_dropin_env_dockerfile() {
  DROPIN_ENV_DIRNAME=$1
  DRUM_WHEEL_REAL_PATH=$2
  DRUM_WHEEL_FILENAME=$(basename "$DRUM_WHEEL_REAL_PATH")
  WITH_R=""
  pwd
  pushd "$DROPIN_ENV_DIRNAME" || exit 1
  cp "$DRUM_WHEEL_REAL_PATH" .

  # check if DRUM is installed with R option
  if grep "datarobot-drum\[R\]" dr_requirements.txt
  then
    WITH_R="[R]"
  fi
  # insert 'COPY wheel wheel' after 'COPY dr_requirements.txt dr_requirements.txt'
  sed -i "/COPY \+dr_requirements.txt \+dr_requirements.txt/a COPY ${DRUM_WHEEL_FILENAME} ${DRUM_WHEEL_FILENAME}" Dockerfile
  # replace 'datarobot-drum' requirement with a wheel
  sed -i "s/^datarobot-drum.*/${DRUM_WHEEL_FILENAME}${WITH_R}/" dr_requirements.txt
  popd || exit 1
}

function build_all_dropin_env_dockerfiles() {
  # Change every environment Dockerfile to install freshly built DRUM wheel
  pushd "${GIT_ROOT}/public_dropin_environments" || exit 1
  DIRS=$(ls)
  for d in $DIRS
  do
    if [[ -d "$d" ]]
    then
       build_dropin_env_dockerfile "$d" "$1"
    fi
  done
  popd || exit 1
}