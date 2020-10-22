#!/usr/bin/env sh
# You probably don't want to modify this file
cd "${CODEPATH}" || exit 1
export PYTHONPATH="${CODEPATH}":"${PYTHONPATH}"

export X="${INPUT_DIRECTORY}/X.csv"
export y="${INPUT_DIRECTORY}/y.csv"
export weights="${INPUT_DIRECTORY}/weights.csv"

TARGET_TYPE="regression"

CMD="drum fit --input ${X} --target-csv ${y} --num-rows ALL --output ${ARTIFACT_DIRECTORY} \
--code-dir ${CODEPATH} --verbose"

if [ -n "${POSITIVE_CLASS_LABEL}" ]; then
    CMD="${CMD} --negative-class-label ${NEGATIVE_CLASS_LABEL} \
    --positive-class-label ${POSITIVE_CLASS_LABEL}"
    TARGET_TYPE="binary"
fi
if [ -n "${CLASS_LABELS_FILE}" ]; then
    CMD="${CMD} --class-labels-file ${CLASS_LABELS_FILE}"
    TARGET_TYPE="multiclass"
if [ -f "${weights}" ]; then
    CMD="${CMD} --row-weights-csv ${weights}"
fi
if [ -n "${TARGET_TYPE}" ]; then
    CMD="${CMD} --target-type ${TARGET_TYPE}"
fi
echo "${CMD}"
sh -c "${CMD}"