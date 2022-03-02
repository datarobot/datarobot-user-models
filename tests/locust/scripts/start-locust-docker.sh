#!/usr/bin/env bash
# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.

# Current script starts locust inside official docker container locustio/locust.
#
# Usage:
# -l - string with native locust params, listed with "locust --help"
# -d - samples dataset to send
# -s - number of samples to send
#
# E.g.
# start-locust-docker.sh -l "-u 5 -r 5 -H http://localhost:6788 -t 5 --headless --csv drum" -d ../testdata/juniors_3_year_stats_regression.csv
#
# locustfile, suit, to use is currently hardcoded to be datarobot-user-models/tests/locust/suits/predict.py

while getopts d:s:l: flag
do
    case "${flag}" in
        d) dataset=${OPTARG};;
        s) samples=${OPTARG};;
        l) locust_params=${OPTARG};;
    esac
done

echo "Provided parameters:"

echo "Dataset: $dataset";
echo "Samples requested: $samples";
echo "Locust params: $locust_params";

START_DIR=$PWD
REPO_DIR=$(git rev-parse --show-toplevel)
IN_DOCKER_REPO_DIR="/mnt/datarobot-user-models"
IN_DOCKER_DATASET_DIR="/mnt/data"
LOCUST_IMAGE_NAME="locustio/locust"

if [ ! -f "${dataset}" ]; then
    echo "Dataset either not provided or doesn't exist."
    echo "Please provide dataset using -d flag"
    exit 1
fi

if [ -n "${samples}" ]; then
    rows_in_file=$(wc -l < ${dataset})
    samples_in_dataset=$(( ${rows_in_file} - 1 ))

    tmp_dataset=$(mktemp /tmp/dataset.XXXXXX)
    if [ ${samples} -gt ${samples_in_dataset} ]; then
        multiplier=$(( ${samples} / ${samples_in_dataset} ))
        cp ${dataset} ${tmp_dataset}
        for (( i=0; i<${multiplier}; i++ ))
        do
            tail +2 ${dataset} >> ${tmp_dataset}
        done
        dataset=${tmp_dataset}
    fi

    # here dataset points to either user provided file or tmp_dataset
    # so need to create another tmp file
    tmp_dataset2=$(mktemp /tmp/dataset.XXXXXX)
    head -$(( ${samples} + 1 )) ${dataset} > $tmp_dataset2

    dataset=${tmp_dataset2}
    rm ${tmp_dataset}
    unset tmp_dataset
fi

echo "Dataset stats (size, name): $(stat -c "%s %n" ${dataset})"
echo "Dataset rows (without header): $(( $(wc -l < ${dataset}) - 1 ))"

DATASET_DIR=$(dirname ${dataset})
DATASET_FILE=$(basename ${dataset})
pushd ${DATASET_DIR} > /dev/null
DATASET_DIR=${PWD}
popd > /dev/null


docker inspect ${LOCUST_IMAGE_NAME} > /dev/null
if [ $? -ne 0 ]
then
   echo "Pulling locust docker image: ${LOCUST_IMAGE_NAME}"
   docker pull ${LOCUST_IMAGE_NAME}
fi

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
        # tmp stub
        echo "Tests are not supported on $machine"
        exit 1
        ;;
    *)
        machine="UNKNOWN:${unameOut}"
        echo "Tests are not supported on $machine"
        exit 1
esac


CMD="docker run --rm 
     --network $network \
     --user $(id -u):$(id -g) \
     -v ${START_DIR}:/home/locust \
     -v ${REPO_DIR}:${IN_DOCKER_REPO_DIR} \
     -v ${DATASET_DIR}:${IN_DOCKER_DATASET_DIR} \
     -e LOCUST_DRUM_DATASET=${IN_DOCKER_DATASET_DIR}/${DATASET_FILE} \
     ${LOCUST_IMAGE_NAME} -f ${IN_DOCKER_REPO_DIR}/tests/locust/suits/predict.py \
     ${locust_params}"

echo "Command to be run: "
echo $CMD
exec ${CMD}
