#!/bin/sh

set -e
ROOT_DIR="$(pwd)"

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

title "Installing pytest"
pip install pytest pytest-xdist

# The FIPS-compliant Java image does not include maven (and its dependencies) required to build Java artifacts
# from source. Therefore, keep the installed dependencies, including datarobot-drum. This means that tests will run
# with the datarobot-drum version specified in the environment's requirements.txt file.
if [ "${FRAMEWORK}" != "java_codegen" ]; then
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


if [ "${FRAMEWORK}" = "vllm" ]; then
    # Note: The `--gpus all` is required for GPU predictors tests
    # Note: For nim_sidecar, the GPUs go to the sidecar container, not the DRUM container
    export GPU_COUNT=$(nvidia-smi -L | wc -l)
    echo "GPU count: $GPU_COUNT"

    GPU_OPTION=""
    if [[ $GPU_COUNT -ge 1 ]] ; then
      GPU_OPTION="--gpus all"
    else
      # Don't set env var if no GPUs are available to tests can be skipped
      unset GPU_COUNT
    fi
fi


pytest ${TESTS_TO_RUN} --framework-env ${FRAMEWORK} --env-folder ${ENV_FOLDER} -rs
