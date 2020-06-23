#!/usr/bin/env bash
# This file will be executed from the root of the repository in a python3 virtualenv.
# It will run the test of drum inside a predefined docker image:

DOCKER_IMAGE="065017677492.dkr.ecr.us-east-1.amazonaws.com/custom_models/cmrun_test_env:2"
CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"
FULL_PATH_CODE_DIR=$(realpath $CDIR)


echo "FULL_PATH_CODE_DIR: $FULL_PATH_CODE_DIR"

echo "Running tests inside docker:"
cd $FULL_PATH_CODE_DIR || exit 1
ls  ./tests/drum/cmrun-tests.sh

unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)
        machine=Linux
        url_host="localhost"
        network="host"
      ;;
    Darwin*)
        machine=Mac
        url_host="host.docker.internal"
        network="bridge"
        ;;
    *)
        machine="UNKNOWN:${unameOut}"
        echo "Tests are not supported on $machine"
        exit 1
esac

# If we are interminal will be true when running the script manually. Via Jenkins it will be false
TERMINAM_OPTION=""
if [ -t 1 ] ; then
  TERMINAM_OPTION="-t"
fi

echo "detected machine=$machine url_host: $url_host"
# Note : The mapping of /tmp is criticall so the code inside the docker can run the tests.
#        Since one of the tests is using a docker the second docker can only share a host file
#        system with the first docker.
# Note: The --network=host will allow a code running inside the docker to access the host network
#       In mac we dont have host network so we use the host.docker.internal ip

docker run -i \
      --network $network \
      -v $HOME:$HOME \
      -e TEST_URL_HOST=$url_host \
      -v /tmp:/tmp \
      -v /var/run/docker.sock:/var/run/docker.sock \
      -v "$FULL_PATH_CODE_DIR:$FULL_PATH_CODE_DIR" \
      --workdir $FULL_PATH_CODE_DIR \
      -i $TERMINAM_OPTION\
      $DOCKER_IMAGE \
      ./tests/drum/cmrun-tests.sh

TEST_RESULT=$?

echo "Done running tests: $TEST_RESULT"
exit $TEST_RESULT
