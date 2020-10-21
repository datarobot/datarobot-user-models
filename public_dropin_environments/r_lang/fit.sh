#!/usr/bin/env sh
# You probably don't want to modify this file
cd "${CODEPATH}" || exit 1
export PYTHONPATH="${CODEPATH}":"${PYTHONPATH}"

export X="${INPUT_DIRECTORY}/X.csv"
export weights="${INPUT_DIRECTORY}/weights.csv"

CMD="drum fit --input ${X} --num-rows ALL --output ${ARTIFACT_DIRECTORY} \
--code-dir ${CODEPATH} --verbose"

if [ -n "${UNSUPERVISED}" ]; then
  CMD="${CMD} --unsupervised "
  TARGET_TYPE="anomaly"
else
  export y="${INPUT_DIRECTORY}/y.csv"
  CMD="${CMD} --target-csv ${y}"
  TARGET_TYPE="regression"
fi


if [ -n "${POSITIVE_CLASS_LABEL}" ]; then
    CMD="${CMD} --negative-class-label ${NEGATIVE_CLASS_LABEL} \
    --positive-class-label ${POSITIVE_CLASS_LABEL}"
    TARGET_TYPE="regression"
fi
if [ -n "${CLASS_LABELS_FILE}" ]; then
    CMD="${CMD} --class-labels-file ${CLASS_LABELS_FILE}"
    TARGET_TYPE="multiclass"
fi
if [ -f "${weights}" ]; then
    CMD="${CMD} --row-weights-csv ${weights}"
fi

if [ -n "${TARGET_TYPE}" ]; then
    CMD="${CMD} --target-type ${TARGET_TYPE}"
fi

echo "${CMD}"
sh -c "${CMD}"
