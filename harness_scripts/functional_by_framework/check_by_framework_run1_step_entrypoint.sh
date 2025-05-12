#!/bin/sh

set -e
ROOT_DIR="$(pwd)"
DRUM_SOURCE_DIR="${ROOT_DIR}/custom_model_runner"

# POSIX compliant way to get the directory of the script
script_dir="${0%/*}"
# If the path is relative, get the absolute path
if [ "${script_dir}" != "${script_dir#/}" ]; then
  script_dir="$(cd "$script_dir" && pwd)"
fi
. ${script_dir}/../common/common.sh

FRAMEWORK=$1
ENV_FOLDER=$2
[ -z ${FRAMEWORK} ] && echo "Environment variable 'FRAMEWORK' is missing" && exit 1
[ -z ${ENV_FOLDER} ] && echo "Environment variable 'ENV_FOLDER' is missing" && exit 1

title "Assuming running integration tests in framework container (inside Docker), for env: '${ENV_FOLDER}/${FRAMEWORK}'"

title "Running as a user:"
# FIPS images don't have id command
set +e
id
set -e

title "Upgrade pip"
pip install -U pip

title "Installing pytest"
pip install pytest pytest-xdist

# The FIPS-compliant Java image does not include maven (and its dependencies) required to build Java artifacts
# from source. Therefore, keep the installed dependencies, including datarobot-drum. This means that tests will run
# with the datarobot-drum version specified in the environment's requirements.txt file.
if [ "${FRAMEWORK}" != "java_codegen" ]; then
    title "Uninstalling datarobot-drum"

    # I think we don't need to uninstall datarobot-mlops
    # pip uninstall datarobot-drum datarobot-mlops -y
    pip uninstall datarobot-drum -y

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

    title "List files in custom_model_runner"
    ls -lah ${DRUM_SOURCE_DIR}

    [ "${FRAMEWORK}" = "r_lang" ] && EXTRA="[R]" || EXTRA=""

    DRUM_SOURCE_DIR_TMP=${DRUM_SOURCE_DIR}

    # GPU envs run not as root, so we can not build within cloned dir, so we copy DRUM source to /tmp
    # FIPS envs dont have cp, but run as root, so we run from cloned folder.
    if [ "${FRAMEWORK}" = "vllm" ]; then
      DRUM_SOURCE_DIR_TMP="/tmp/custom_model_runner"
      cp -r ${DRUM_SOURCE_DIR} ${DRUM_SOURCE_DIR_TMP}
    fi
    # Install datarobot-drum from source code.
    # Testing image either was just built (if env changed), or the latest release image is used for testing.
    # I think we should not reinstall all the deps.
    # We only install DRUM from source, if some deps were changed they are upgraded.
    # This saves time by avoiding heavy AI/nvidia packages reinstallation.
    # pip install --force-reinstall ${DRUM_SOURCE_DIR_TMP}${EXTRA} ${INST_ENV_REQ_CMD}
    pip install --upgrade ${DRUM_SOURCE_DIR_TMP}${EXTRA}
fi

cd "${ROOT_DIR}"
TESTS_TO_RUN="tests/functional/test_inference_per_framework.py \
              tests/functional/test_inference_gpu_predictors.py \
              tests/functional/test_fit_per_framework.py \
              tests/functional/test_other_cases_per_framework.py \
              tests/functional/test_unstructured_mode_per_framework.py
             "

title "Start testing"
if [ "${FRAMEWORK}" = "r_lang" ]; then
    Rscript -e "install.packages('pack', repos='https://cloud.r-project.org', Ncpus=4)"
    TESTS_TO_RUN="${TESTS_TO_RUN} tests/integration/datarobot_drum/drum/language_predictors/test_language_predictors.py::TestRPredictor \
                   tests/unit/datarobot_drum/drum/utils/test_drum_utils.py \
                   tests/unit/datarobot_drum/model_metadata/test_model_metadata.py
                  "
fi

pytest ${TESTS_TO_RUN} --framework-env ${FRAMEWORK} --env-folder ${ENV_FOLDER} -rs
