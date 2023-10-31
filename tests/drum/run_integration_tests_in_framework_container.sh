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


echo "-- Assuming running integration tests in framework container (inside Docker), for env: $1"
echo "Installing pytest"

pip install pytest pytest-xdist


TESTS_TO_RUN="tests/drum/test_inference_per_framework.py \
              tests/drum/test_fit_per_framework.py \
              tests/drum/test_other_cases_per_framework.py \
              tests/drum/test_unstructured_mode_per_framework.py
             "

# install 'pack' package in R env for tests
if [ "$1" = "r_lang" ]; then
    Rscript -e "install.packages('pack', Ncpus=4)"
    TESTS_TO_RUN+="tests/integration/datarobot_drum/drum/language_predictors/test_language_predictors.py::TestRPredictor \
                   tests/unit/datarobot_drum/drum/utils/test_drum_utils.py \
                   tests/unit/datarobot_drum/model_metadata/test_model_metadata.py
                  "
fi

pytest ${TESTS_TO_RUN} \
       --framework-env $1 \
       --junit-xml="./results_integration.xml" \
       -n auto

TEST_RESULT=$?


if [ $TEST_RESULT -ne 0 ] ; then
  echo "Got error in one of the tests"
  echo "Rest of tests: $TEST_RESULT"
  exit 1
else
  exit 0
fi
