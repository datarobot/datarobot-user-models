#!/usr/bin/env bash
# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.

echo "--- env ----"
export
echo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


echo "-- running drum tests - assuming running inside Docker"

GIT_ROOT=$(git rev-parse --show-toplevel)
echo "GIT_ROOT: $GIT_ROOT"
echo
echo "Running pytest:"

pip install pytest pytest-xdist


TESTS_TO_RUN="tests/drum/test_inference_per_framework.py \
              tests/drum/test_fit_per_framework.py \
              tests/drum/test_other_cases_per_framework.py \
              tests/drum/test_unstructured_mode_per_framework.py
             "

# install 'pack' package in R env for tests
if [ "$1" = "r_lang" ]; then
    Rscript -e "install.packages('pack', Ncpus=4)"
    TESTS_TO_RUN+="tests/drum/unit/test_language_predictors.py \
                   tests/drum/unit/test_utils.py \
                   tests/drum/unit/model_metadata/test_model_metadata.py
                  "
fi

pytest ${TESTS_TO_RUN} \
       --framework-env $1 \
       --junit-xml="$GIT_ROOT/results_integration.xml" \
       -n auto

TEST_RESULT=$?


if [ $TEST_RESULT -ne 0 ] ; then
  echo "Got error in one of the tests"
  echo "Rest of tests: $TEST_RESULT"
  exit 1
else
  exit 0
fi
