#!/bin/bash

# we don't want it output anything in the terminal session setup
VERBOSE_MODE=${1:-false}

IS_CODESPACE=$([[ "${WORKING_DIR}" == *"/storage"* ]] && echo true || echo false)
IS_PYTHON_KERNEL=$([[ "${NOTEBOOKS_KERNEL}" == "python" ]] && echo true || echo false)

if [[ $IS_CODESPACE == true ]]; then
  # set global variables for all kernels (python, R, etc.) in codespaces
  export XDG_CACHE_HOME="${WORKING_DIR%/}/.cache"
  export XDG_CONFIG_HOME="${WORKING_DIR%/}/.config"
  export XDG_CONFIG_DIRS="${HOME}/.config"
  export COLORTERM=truecolor
fi

if [[ $IS_CODESPACE == true && $IS_PYTHON_KERNEL == true && -z "${NOTEBOOKS_NO_PERSISTENT_DEPENDENCIES}" ]]; then
  export POETRY_VIRTUALENVS_CREATE=false
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

  # If a user has previously created a session with a different python version we need to figure that out
  # If so we'll delete the existing venv to avoid errors and issues - for example when pip installing new packages
  if [ -d "$USR_VENV" ]; then
    [[ $VERBOSE_MODE == true ]] && echo "$USR_VENV does exist - will check python symlinks to see if they are broken..."
    # Here we are getting all the symlinks for the venv and checking if any of them are broken
    readarray -d '' VENV_SYMLINKS < <(find "$USR_VENV" -type l -print0)
    python_symlinks_broken=false
    for i in "${VENV_SYMLINKS[@]}"; do
      if [[ "$i" == *"python"* ]]; then
        [[ $VERBOSE_MODE == true ]] && echo "Checking symlink (${i}).";
        if [ ! -e "$i" ] ; then
          [[ $VERBOSE_MODE == true ]] && echo "Symlink (${i}) broken...";
          python_symlinks_broken=true
          break
        fi
      fi
    done

    # If any python symlinks are broken delete the venv that we know exists from checks above
    if [[ $python_symlinks_broken == true ]]; then
      [[ $VERBOSE_MODE == true ]] && echo "Python symlinks are broken - deleting existing virtual env..."
      rm -rf "${USR_VENV}"
    fi
  fi

  python3 -m venv "${USR_VENV}"
  # shellcheck disable=SC1091
  source "${USR_VENV}/bin/activate"
  USER_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

  export PYTHONPATH="$USER_PACKAGES:$KERNEL_PACKAGES:$PYTHONPATH"
elif [[ $IS_PYTHON_KERNEL == true ]]; then
  # FLEET-8918: the kernel venv is writable only at the image's own uid, so give these sessions
  # the same thin user venv codespaces get, built only once that venv is proven unwritable so a
  # default install and anything a customer layered in stay untouched (narrower than nbx-kernels#390).
  # shellcheck disable=SC1091
  source "$VENV_PATH/bin/activate"
  KERNEL_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

  if [[ -w "$KERNEL_PACKAGES" ]]; then
    [[ $VERBOSE_MODE == true ]] && echo "Kernel venv is writable; skipping user venv setup..."
  else
    deactivate
    USR_VENV="${NOTEBOOKS_RW_DIR:-/home/notebooks/.nbx-rw}/venv"
    [[ $VERBOSE_MODE == true ]] && echo "Kernel venv is read-only at this uid; setting up a user venv ($USR_VENV)..."

    # setup-prompt.sh sources this file from /etc/profile.d, so the venv is only built when it
    # is not already there and a terminal login reuses the one the session started with.
    if [ -x "${USR_VENV}/bin/python" ] || python3 -m venv "${USR_VENV}" 2>/dev/null; then
      # shellcheck disable=SC1091
      source "${USR_VENV}/bin/activate"
      USER_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
      # A .pth rather than PYTHONPATH alone, so the kernel venv's own .pth files are processed
      # and anything installed there as editable keeps resolving.
      echo "import site; site.addsitedir('${KERNEL_PACKAGES}')" > "${USER_PACKAGES}/zz-nbx-kernel-venv.pth"
      export PYTHONPATH="$USER_PACKAGES:$KERNEL_PACKAGES:$PYTHONPATH"

      # jupyter_client rewrites a kernelspec argv[0] of "python" to whichever interpreter the
      # gateway runs on, which is the read-only kernel venv, so a cell's pip install would still
      # fail; name the user venv in a kernelspec of our own and put it first on JUPYTER_PATH.
      NBX_KERNELS_DIR="${JUPYTER_DATA_DIR:-${NOTEBOOKS_RW_DIR:-/home/notebooks/.nbx-rw}/jupyter/data}/kernels/python3"
      if mkdir -p "$NBX_KERNELS_DIR" 2>/dev/null && python -c 'import json,sys; s=json.load(open(sys.argv[1])); s["argv"][0]=sys.argv[3]; json.dump(s, open(sys.argv[2],"w"), indent=2)' \
          "${VENV_PATH}/share/jupyter/kernels/python3/kernel.json" "${NBX_KERNELS_DIR}/kernel.json" "${USR_VENV}/bin/python"; then
        export JUPYTER_PATH="${NBX_KERNELS_DIR%/kernels/python3}"
      else
        echo "WARNING: could not write a kernelspec at ${NBX_KERNELS_DIR}, so installing packages from a cell will still fail." >&2
      fi
    else
      echo "WARNING: could not create a user venv at ${USR_VENV}; falling back to the kernel venv." >&2
      echo "WARNING: installing packages from a cell will fail unless this session runs as the image's own uid." >&2
      # shellcheck disable=SC1091
      source "$VENV_PATH/bin/activate"
    fi
  fi
else
  [[ $VERBOSE_MODE == true ]] && echo "Skipping user venv setup (not a python kernel)..."
  # shellcheck disable=SC1091
  source "$VENV_PATH/bin/activate"
fi
