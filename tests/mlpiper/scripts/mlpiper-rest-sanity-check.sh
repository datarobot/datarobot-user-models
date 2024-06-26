#!/usr/bin/env bash

# This script is designed to perform a RESTful component sanity-check for 'mlpiper'.
#
# Here are the steps that are executed:
#   1. Check that wheel has been built. It is, if started with make sanity_restful,
#      otherwise build it manually: make dist
#   2. Docker image datarobot/python3-nginx-mlpiper-test, containing nginx, is pulled
#   3. Docker container is started and start-mlpiper-with-rest-inside-docker.sh script is executed inside:
#      a. mlpiper wheel is installed
#      b. Dependencies are installed
#      c. mlpiper started (starts nginx, uwsgi)
#   4. Waiting for container to start and server to response
#   5. Two requests are submitted with curl and result checked

SCRIPT_NAME=$(basename ${BASH_SOURCE[0]})
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MLPIPER_ROOT=$(git rev-parse --show-toplevel)
MLPIPER_PY_ROOT=${MLPIPER_ROOT}/mlpiper-py

MLPIPER_WHEEL=$(find ${MLPIPER_PY_ROOT}/dist/mlpiper*-py2.py3-*.whl)
if [ $? -ne 0 ]; then
  echo "Can't find mlpiper wheel file in mlpiper-py/dist. Build it or do: make sanity_restful"
  exit 1
fi

docker run -id -p 8888:8888 -v ${MLPIPER_ROOT}:/tmp/mlpiper datarobot/python3-nginx-mlpiper-test /tmp/mlpiper/mlpiper-py/tests/scripts/start-mlpiper-with-rest-inside-docker.sh

counter=0
timeout=200
while [[ $counter -lt $timeout ]]
do
  curl -s localhost:8888/statsinternal/ > /dev/null
  if [ $? -eq 0 ]; then
    break
  fi
  echo "Waiting for server to start in a docker container... $counter/$timeout"
  sleep 1
  ((counter++))
done

test_passed=127
value1=$(curl -s -d '{"data": [1,2,3,4,5,6,7,8,9,10,11]}' -H "Content-Type: application/json" -X POST localhost:8888/predict | python3 -c "import json, sys; print(json.load(sys.stdin)['prediction'])")
value2=$(curl -s -d '{"data": [33,2,3,4,5,6,7,8,9,10,11]}' -H "Content-Type: application/json" -X POST localhost:8888/predict | python3 -c "import json, sys; print(json.load(sys.stdin)['prediction'])")
if [ $value1 -eq 2 ] && [ $value2 -eq 3 ]; then
  test_passed=0
fi

docker rm --force $(docker ps -lq) > /dev/null


if [ $test_passed -eq 0 ]; then
  echo "mlpiper RESTful sanity check has PASSED!"
else
  echo "mlpiper RESTful sanity check has FAILED!"
fi
exit $test_passed