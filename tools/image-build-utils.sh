#!/usr/bin/env bash
# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.

GIT_ROOT=$(git rev-parse --show-toplevel)

function build_drum() {
  CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"
  DRUM_BUILDER_IMAGE="datarobot/drum-builder:ubuntu-22-04"

  # pull DRUM builder container and build DRUM wheel
  docker pull ${DRUM_BUILDER_IMAGE}

  # If we are in terminal will be true when running the script manually. Via Jenkins it will be false.
  TERMINAL_OPTION=""
  if [ -t 1 ] ; then
    TERMINAL_OPTION="-t"
  fi

  docker run -i ${TERMINAL_OPTION} -v $CDIR:/tmp/drum $DRUM_BUILDER_IMAGE bash -c "cd /tmp/drum/custom_model_runner && make"
  docker rmi $DRUM_BUILDER_IMAGE --force

}

function build_dropin_env_dockerfile() {
  DROPIN_ENV_DIRNAME=$1
  DRUM_WHEEL_REAL_PATH=$2
  DRUM_WHEEL_FILENAME=$(basename "$DRUM_WHEEL_REAL_PATH")
  WITH_R=""

  if [ "$DROPIN_ENV_DIRNAME" = "python39_streamlit" ] || [ "$DROPIN_ENV_DIRNAME" = "python312_apps" ]; then
    return 0
  fi

  pwd
  pushd "$DROPIN_ENV_DIRNAME" || exit 1
  cp "$DRUM_WHEEL_REAL_PATH" .

  # check if DRUM is installed with R option
  if grep "datarobot-drum\[R\]" requirements.txt
  then
    WITH_R="[R]"
  fi

  # support Darwin
  if command -v gsed &> /dev/null; then
    local sed=gsed
  else
    local sed=sed
  fi
  # insert 'COPY wheel wheel' after 'COPY requirements.txt requirements.txt'
  if ! grep -q "COPY \+${DRUM_WHEEL_FILENAME}" Dockerfile; then
    $sed -i "/COPY \+requirements.txt \+requirements.txt/a COPY ${DRUM_WHEEL_FILENAME} ${DRUM_WHEEL_FILENAME}" Dockerfile
  fi
  # replace 'datarobot-drum' requirement with a wheel
  $sed -i "s/^datarobot-drum.*/${DRUM_WHEEL_FILENAME}${WITH_R}/" requirements.txt

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
