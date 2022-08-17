#!/usr/bin/env bash
# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
# This file will be executed from the root of the repository in a python3 virtualenv.
# It will run the test of drum inside a predefined docker image:

# root repo folder
GIT_ROOT=$(git rev-parse --show-toplevel)
echo "GIT_ROOT: $GIT_ROOT"


# Installing and configuring java/javac 11 in the jenkins worker
sudo apt update
sudo apt install --no-install-recommends -y openjdk-11-jdk openjdk-11-jre zip
sudo update-alternatives --set java /usr/lib/jvm/java-11-openjdk-amd64/bin/java
sudo update-alternatives --auto javac
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

echo "DEBUG: print java version"
java -version
javac -version


pip install -U pip
pip install pytest pytest-runner pytest-xdist retry

# > NOTE: when pinning datarobot-mlops to 8.2.1 and higher you may need to reinstall datarobot package
# as datarobot-mlops overwrites site-packages/datarobot. [AGENT-3504]
pip install datarobot-mlops==8.1.3


pushd ${GIT_ROOT} || exit 1

DRUM_WHEEL_REAL_PATH="$(realpath "$(find jenkins_artifacts/datarobot_drum*.whl)")"
echo
echo "--> Installing wheel: ${DRUM_WHEEL_REAL_PATH}"
echo
pip install "${DRUM_WHEEL_REAL_PATH}"

echo
echo "--> Compiling jar for TestCustomPredictor "
echo
pushd $GIT_ROOT/tests/drum/custom_java_predictor
mvn package
popd


# only run here tests which were sequential historically

pytest tests/drum/test_inference_custom_java_predictor.py tests/drum/test_mlops_monitoring.py \
       -m "sequential" \
       --junit-xml="$GIT_ROOT/results_integration.xml" \
       -n 1

TEST_RESULT_1=$?

popd

if [ $TEST_RESULT_1 -ne 0 ] ; then
  echo "Got error in one of the tests"
  echo "Sequential tests: $TEST_RESULT_1"
  exit 1
else
  exit 0
fi

