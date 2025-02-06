#!/bin/bash

set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. ${script_dir}/../common/common.sh

FRAMEWORK=$1
[ -z $FRAMEWORK ] && echo "Environment variable 'FRAMEWORK' is missing" && exit 1

title "Assuming running integration tests in framework container (inside Docker), for env: '$FRAMEWORK'"

ROOT_DIR="$(pwd)"
PUBLIC_ENVS_DIR="${ROOT_DIR}/public_dropin_environments"
ENV_REQ_FILE_PATH="${PUBLIC_ENVS_DIR}/${FRAMEWORK}/requirements.txt"
[ ! -f $ENV_REQ_FILE_PATH ] && echo "Requirements file not found: $ENV_REQ_FILE_PATH" && exit 1

script_dir=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
. ${script_dir}/../../tools/update-python-to-meet-requirements.sh

title "Installing pytest"
pip install pytest pytest-xdist

title "Uninstalling datarobot-drum"
pip uninstall datarobot-drum datarobot-mlops -y

title "Installing dependencies, with datarobot-drum installed from source-code"

pushd custom_model_runner

if [ "$FRAMEWORK" = "java_codegen" ]; then
    make java_components
fi

temp_requirements_file=$(mktemp)
cp ${ENV_REQ_FILE_PATH} ${temp_requirements_file}
# remove DRUM from requirements file to be able to install it from source
sed -i "s/^datarobot-drum.*//" ${temp_requirements_file}

[ "$FRAMEWORK" = "r_lang" ] && EXTRA="[R]" || EXTRA=""
pip install --force-reinstall -r ${temp_requirements_file} .$EXTRA
rm -rf build datarobot_drum.egg-info dist ${temp_requirements_file}

popd  # custom_model_runner

TESTS_TO_RUN="tests/functional/test_inference_per_framework.py \
              tests/functional/test_fit_per_framework.py \
              tests/functional/test_other_cases_per_framework.py \
              tests/functional/test_unstructured_mode_per_framework.py
             "

title "Start testing"
if [ "$FRAMEWORK" = "r_lang" ]; then
    Rscript -e "install.packages('pack', Ncpus=4)"
    TESTS_TO_RUN+="tests/integration/datarobot_drum/drum/language_predictors/test_language_predictors.py::TestRPredictor \
                   tests/unit/datarobot_drum/drum/utils/test_drum_utils.py \
                   tests/unit/datarobot_drum/model_metadata/test_model_metadata.py
                  "
fi

pytest ${TESTS_TO_RUN} --framework-env $FRAMEWORK
