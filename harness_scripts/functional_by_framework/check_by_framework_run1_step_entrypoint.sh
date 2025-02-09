#!/bin/sh

set -e
ROOT_DIR="$(pwd)"

script_dir="$(cd "$(dirname "$0")" && pwd)"
. ${script_dir}/../common/common.sh

FRAMEWORK=$1
ENV_FOLDER=$2
[ -z ${FRAMEWORK} ] && echo "Environment variable 'FRAMEWORK' is missing" && exit 1
[ -z ${ENV_FOLDER} ] && echo "Environment variable 'ENV_FOLDER' is missing" && exit 1

title "Assuming running integration tests in framework container (inside Docker), for env: '${ENV_FOLDER}/${FRAMEWORK}'"

title "Installing pytest"
pip install pytest pytest-xdist

title "Uninstalling datarobot-drum"
pip uninstall datarobot-drum datarobot-mlops -y

title "Installing dependencies, with datarobot-drum installed from source-code"

PUBLIC_ENV_PATH="${ROOT_DIR}/${ENV_FOLDER}/${FRAMEWORK}"
ENV_REQ_FILE_PATH="${PUBLIC_ENV_PATH}/requirements.txt"
if [ -f ${ENV_REQ_FILE_PATH} ]; then
    # Make sure to install the requirements from the environment, but without datarobot-drum
    cd "${PUBLIC_ENV_PATH}/"
    TMP_ENV_REQ_FILE_PATH=/tmp/requirements.txt
    python3 -c "print('\n'.join([line for line in open('requirements.txt') if not line.startswith('datarobot-drum')]))"  > ${TMP_ENV_REQ_FILE_PATH}
    INST_ENV_REQ_CMD=" -r ${TMP_ENV_REQ_FILE_PATH}"
else
    echo "No requirements file found at: ${ENV_REQ_FILE_PATH}"
    INST_ENV_REQ_CMD=""
fi

cd "${ROOT_DIR}/custom_model_runner"
if [ "${FRAMEWORK}" = "java_codegen" ]; then
    make java_components
fi
[ "${FRAMEWORK}" = "r_lang" ] && EXTRA="[R]" || EXTRA=""
# Install datarobot-drum from source code, but keep dependencies that were installed by the environment
pip install --force-reinstall .${EXTRA} ${INST_ENV_REQ_CMD}

cd "${ROOT_DIR}"
TESTS_TO_RUN="tests/functional/test_inference_per_framework.py \
              tests/functional/test_fit_per_framework.py \
              tests/functional/test_other_cases_per_framework.py \
              tests/functional/test_unstructured_mode_per_framework.py
             "

title "Start testing"
if [ "${FRAMEWORK}" = "r_lang" ]; then
    Rscript -e "install.packages('pack', Ncpus=4)"
    TESTS_TO_RUN="${TESTS_TO_RUN} tests/integration/datarobot_drum/drum/language_predictors/test_language_predictors.py::TestRPredictor \
                   tests/unit/datarobot_drum/drum/utils/test_drum_utils.py \
                   tests/unit/datarobot_drum/model_metadata/test_model_metadata.py
                  "
fi

pytest ${TESTS_TO_RUN} --framework-env ${FRAMEWORK} --env-folder ${ENV_FOLDER}
