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

function create_virtual_env() {
    venv=${1}
    py_ver=${2}

    $(venv_exists_check ${venv}) && { title "Removing '${venv}' venv ..."; rmvirtualenv ${venv}; }

    title "Creating '${venv}' ..."
    mkvirtualenv --python=${py_ver} ${venv}

    [[ "$VIRTUAL_ENV" == "" ]] && { title "Failed to create: '${venv}'! Exiting ..."; exit -1; }
    pip install wheel

    deactivate
}

function install_wheel_into_venv() {
    wheel=${1}
    venv=${2}

    workon ${venv}
    title "Reinstall ${wheel} wheel on '${venv}' ..."
    pip install -U ${wheel} || { title "Failed installing '${wheel}' in '${venv}'! Exiting ..."; exit -1; }
    deactivate
}
