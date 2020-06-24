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

function build_docker_image_with_cmrun() {
  orig_docker_context_dir=$1
  image_name=$2
  drum_wheel=$3
  drum_requirements=$4

  docker_dir=/tmp/cmrun_docker.$$

  echo "Building docker image:"
  echo "orig_docker_context_dir: $orig_docker_context_dir"
  echo "image_name:              $image_name"
  echo "drum_wheel:              $drum_wheel"
  echo "drum_requirements:       $drum_requirements"
  echo "docker_dir:              $docker_dir"

  rm -rf $docker_dir
  cp -a $orig_docker_context_dir $docker_dir

  cp $drum_wheel $docker_dir
  cp $drum_requirements $docker_dir/drum_requirements.txt

  pushd $docker_dir || exit 1
  docker build -t $image_name ./
  popd
  rm -rf $docker_dir
  echo
  echo
  docker images
}