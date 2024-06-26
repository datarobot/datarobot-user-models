function title() {
    msg=$1
    GREEN='\033[1;32m'
    NC='\033[0m' # No Color

    echo
    echo -e "*** ${GREEN}$msg${NC}"
}

function venv_exists_check() {
    venv=$1

    lsvirtualenv -b | grep "^${venv}$" > /dev/null
}

function create_mlpiper_virtual_env() {
    venv=${1}
    py_ver=${2}
    wheel_file=${3}

    $(venv_exists_check ${venv}) && { title "Removing '${venv}' venv ..."; rmvirtualenv ${venv}; }

    title "Creating '${venv}' ..."
    mkvirtualenv --python python${py_ver} ${venv}

    [[ "$VIRTUAL_ENV" == "" ]] && { title "Failed to create: '${venv}'! Exiting ..."; exit -1; }
    pip install wheel

    title "Installling mlpiper wheel on '${venv}' ..."
    cd $MLPIPER_PY_ROOT
    pip install ${wheel_file} || { title "Failed installing 'mlpiper' in '${venv}'! Exiting ..."; exit -1; }

    deactivate
}

function setup_xargs_tool() {
    if [[ $OSTYPE =~ ^darwin.* ]]; then
      if ! $(gxargs --version &> /dev/null); then
          brew install findutils &> /dev/null
      fi

      echo 'gxargs'
    else
      echo 'xargs'
    fi
}
