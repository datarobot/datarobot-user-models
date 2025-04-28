#!/bin/bash

# we don't want it output anything in the terminal session setup
VERBOSE_MODE=${1:-false}

IS_CODESPACE=$([[ "${WORKING_DIR}" == *"/storage"* ]] && echo true || echo false)
IS_PYTHON_KERNEL=$([[ "${NOTEBOOKS_KERNEL}" == "python" ]] && echo true || echo false)

if [[ $IS_CODESPACE == true && $IS_PYTHON_KERNEL == true && -z "${NOTEBOOKS_NO_PERSISTENT_DEPENDENCIES}" ]]; then
  export POETRY_VIRTUALENVS_CREATE=false
  export XDG_CACHE_HOME="${WORKING_DIR%/}/.cache"
  # Persistent HF artifact installation
  export HF_HOME="${WORKING_DIR%/}/.cache"
  export HF_HUB_CACHE="${WORKING_DIR%/}/.cache"
  export HF_DATASETS_CACHE="${WORKING_DIR%/}/.datasets"
  export TRANSFORMERS_CACHE="${WORKING_DIR%/}/.models"
  export SENTENCE_TRANSFORMERS_HOME="${WORKING_DIR%/}/.models"

  USR_VENV="${WORKING_DIR%/}/.venv"
  [[ $VERBOSE_MODE == true ]] && echo "Setting up a user venv ($USR_VENV)..."

  # we need to make sure both kernel & user venv's site-packages are in PYTHONPATH because:
  # - when the user venv is activated (e.g. terminal sessions), it ignores the kernel venv
  # - when Jupyter kernel is running (e.g. notebook cells) it uses the kernel venv ignoring the user venv

  # shellcheck disable=SC1091
  source "$VENV_PATH/bin/activate"
  KERNEL_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
  deactivate

  python3 -m venv "${USR_VENV}"
  # shellcheck disable=SC1091
  source "${USR_VENV}/bin/activate"
  USER_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

  export PYTHONPATH="$USER_PACKAGES:$KERNEL_PACKAGES:$PYTHONPATH"
else
  [[ $VERBOSE_MODE == true ]] && echo "Skipping user venv setup..."
  # shellcheck disable=SC1091
  source "$VENV_PATH/bin/activate"
fi