#!/usr/bin/env bash
# This file will be executed from the root of the repository in a python3 virtualenv.
# It will run the test of drum inside a predefined docker image:

set -ex

DOCKER_IMAGE="datarobot/drum_integration_tests_base"
CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"
FULL_PATH_CODE_DIR=$(realpath $CDIR)


echo "FULL_PATH_CODE_DIR: $FULL_PATH_CODE_DIR"

echo "Running tests inside docker:"
cd $FULL_PATH_CODE_DIR || exit 1
ls  ./tests/drum/run-drum-tests-in-container.sh

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

# If we are in terminal will be true when running the script manually. Via Jenkins it will be false
TERMINAM_OPTION=""
if [ -t 1 ] ; then
  TERMINAM_OPTION="-t"
fi

echo "detected machine=$machine url_host: $url_host"
# Note : The mapping of /tmp is critical so the code inside the docker can run the tests.
#        Since one of the tests is using a docker the second docker can only share a host file
#        system with the first docker.
# Note: The --network=host will allow a code running inside the docker to access the host network
#       In mac we dont have host network so we use the host.docker.internal ip

TEST_URL_HOST=$url_host

./tests/drum/run-drum-tests-in-container.sh

TEST_RESULT=$?

echo "Done running tests: $TEST_RESULT"
exit $TEST_RESULT
