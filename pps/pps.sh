#!/bin/sh

function title() {
    msg=$1
    GREEN='\033[1;32m'
    NC='\033[0m' # No Color

    echo
    echo -e "*** ${GREEN}$msg${NC}"
}



SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TMP_DIR=${SCRIPT_DIR}

ENV_ARCHIVE_NAME=$(python3 -c "import sys, json; print(json.load(open('manifest.json'))['env'])")
MODEL_ARCHIVE_NAME=$(python3 -c "import sys, json; print(json.load(open('manifest.json'))['model'])")
MLOPS_ARCHIVE_NAME=$(python3 -c "import sys, json; print(json.load(open('manifest.json'))['mlops'])")

DEPS_INSTALL_SCRIPT=requirements_install.sh

ENV_EXTRACT_DIR="${TMP_DIR}/env"
MODEL_EXTRACT_DIR="${TMP_DIR}/model"
MLOPS_EXTRACT_DIR="${TMP_DIR}/mlops"

rm -rf ${ENV_EXTRACT_DIR} ${MODEL_EXTRACT_DIR} ${MLOPS_EXTRACT_DIR}

title "Extracting artifacts"
mkdir -p ${ENV_EXTRACT_DIR}
tar -xvf ${ENV_ARCHIVE_NAME} -C ${ENV_EXTRACT_DIR}

mkdir -p ${MLOPS_EXTRACT_DIR}
tar -xvf ${MLOPS_ARCHIVE_NAME} -C ${MLOPS_EXTRACT_DIR}

mkdir -p ${MODEL_EXTRACT_DIR}
unzip ${MODEL_ARCHIVE_NAME} -d ${MODEL_EXTRACT_DIR}

MLOPS_WHEEL=$(find ${MLOPS_EXTRACT_DIR} -name datarobot_mlops*.whl)


title "Building environment docker image"
pushd ${ENV_EXTRACT_DIR}


# copy all the files from model
cp -r ${MODEL_EXTRACT_DIR}/* .
# copy mlops wheel
cp ${MLOPS_WHEEL} .
cp Dockerfile Dockerfile.bak
MLOPS_WHEEL_BASENAME=$(basename ${MLOPS_WHEEL})

# add copy and install mlops commands into dockerfile
echo "COPY ${MLOPS_WHEEL_BASENAME} ${MLOPS_WHEEL_BASENAME}" >> Dockerfile
echo "RUN pip3 install ${MLOPS_WHEEL_BASENAME}" >> Dockerfile
# remove once newer mlops package is released
echo "RUN pip3 install py4j==0.10.9.1" >> Dockerfile
if test -f ${DEPS_INSTALL_SCRIPT}; then
    echo "COPY ${DEPS_INSTALL_SCRIPT} ${DEPS_INSTALL_SCRIPT}" >> Dockerfile
    echo "RUN chmod +x ${DEPS_INSTALL_SCRIPT}" >> Dockerfile
    echo "RUN ./${DEPS_INSTALL_SCRIPT}" >> Dockerfile
fi

# to do: name docker image" pps_env+deployment_d
docker build -t pps_env .

popd



title "Building mlops agent docker image"
pushd ${MLOPS_EXTRACT_DIR}
pushd $(find . -name datarobot_mlops_package*)
pushd "tools/agent_docker/"
echo $(pwd)
. ./run.sh build

popd
popd
popd



