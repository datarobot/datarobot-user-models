#!/usr/bin/env sh
# You probably don't want to modify this file
cd "${CODEPATH}" || exit 1
export PYTHONPATH="${CODEPATH}":"${PYTHONPATH}"

export X="${INPUT_DIRECTORY}/X${TRAINING_DATA_EXTENSION:-.csv}"
export weights="${INPUT_DIRECTORY}/weights.csv"
export sparse_colnames="${INPUT_DIRECTORY}/X.colnames"

CMD="drum fit --target-type ${TARGET_TYPE} --input ${X} --num-rows ALL --output ${ARTIFACT_DIRECTORY} \
--code-dir ${CODEPATH} --verbose"

if [ "${TARGET_TYPE}" != "anomaly" ]; then
    CMD="${CMD} --target-csv ${INPUT_DIRECTORY}/y.csv"
fi

if [ -f "${weights}" ]; then
    CMD="${CMD} --row-weights-csv ${weights}"
fi

if [ -f "${sparse_colnames}" ]; then
    CMD="${CMD} --sparse-column-file ${sparse_colnames}"
fi

echo "Environment variables:"
env
echo "${CMD}"
sh -c "${CMD}"
