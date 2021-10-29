#!/usr/bin/env sh
# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
# You probably don't want to modify this file
cd "${CODEPATH}" || exit 1
export PYTHONPATH="${CODEPATH}":"${PYTHONPATH}"

export X="${INPUT_DIRECTORY}/X${TRAINING_DATA_EXTENSION:-.csv}"
export weights="${INPUT_DIRECTORY}/weights.csv"
export offsets="${INPUT_DIRECTORY}/offsets.csv"
export events_count="${INPUT_DIRECTORY}/events_count.csv"
export sparse_colnames="${INPUT_DIRECTORY}/X.colnames"
export parameters="${INPUT_DIRECTORY}/parameters.json"

CMD="drum fit --target-type ${TARGET_TYPE} --input ${X} --num-rows ALL --output ${ARTIFACT_DIRECTORY} \
--code-dir ${CODEPATH} --verbose"


if [ "${TARGET_TYPE}" != "anomaly" ]; then
    CMD="${CMD} --target-csv ${INPUT_DIRECTORY}/y.csv"
fi

if [ -f "${weights}" ]; then
    CMD="${CMD} --row-weights-csv ${weights}"
fi

if [ -f "${offsets}" ]; then
    CMD="${CMD} --offsets-csv ${offsets}"
fi

if [ -f "${events_count}" ]; then
    CMD="${CMD} --events-count-csv ${events_count}"
fi

if [ -f "${sparse_colnames}" ]; then
    CMD="${CMD} --sparse-column-file ${sparse_colnames}"
fi

if [ -f "${parameters}" ]; then
    CMD="${CMD} --parameter-file ${parameters}"
fi

echo "Environment variables:"
env
echo "${CMD}"
sh -c "${CMD}"
