#!/bin/bash

FRAMEWORK=$1
[ -z $FRAMEWORK ] && echo "Environment variable 'FRAMEWORK' is missing" && exit 1

echo "== Assuming running integration tests in framework container (inside Docker), for env: <+pipeline.variables.framework>"
ROOT_DIR="$(pwd)"


echo "== Installing pytest =="
pip install pytest pytest-xdist
echo "== Uninstalling datarobot-drum =="
pip uninstall datarobot-drum -y
cd custom_model_runner
echo "== Install datarobot-drum from source =="
if [ "$FRAMEWORK" = "java_codegen" ]; then
    make java_components
fi

PUBLIC_ENVS_DIR="${ROOT_DIR}/public_dropin_environments"
REQ_FILE_PATH="${PUBLIC_ENVS_DIR}/${FRAMEWORK}/requirements.txt"

# remove DRUM from requirements file to be able to install it from source
sed -i "s/^datarobot-drum.*//" ${REQ_FILE_PATH}

pip install --force-reinstall -r ${REQ_FILE_PATH} .

cd -
TESTS_TO_RUN="tests/functional/test_inference_per_framework.py \
              tests/functional/test_fit_per_framework.py \
              tests/functional/test_other_cases_per_framework.py \
              tests/functional/test_unstructured_mode_per_framework.py
             "

if [ "$FRAMEWORK" = "r_lang" ]; then
    Rscript -e "install.packages('pack', Ncpus=4)"
    TESTS_TO_RUN+="tests/integration/datarobot_drum/drum/language_predictors/test_language_predictors.py::TestRPredictor \
                   tests/unit/datarobot_drum/drum/utils/test_drum_utils.py \
                   tests/unit/datarobot_drum/model_metadata/test_model_metadata.py
                  "
fi

pytest ${TESTS_TO_RUN} --framework-env $FRAMEWORK