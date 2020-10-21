#!/bin/sh
cd /opt/code/ || exit 1
export PYTHONPATH=/opt/code

TARGET_TYPE="regression"

CMD="drum server -cd . --address 0.0.0.0:8080 --production --max-workers 1 --show-stacktrace"

if [ -n "${POSITIVE_CLASS_LABEL}" ]; then
    CMD="${CMD} --positive-class-label ${POSITIVE_CLASS_LABEL}"
    TARGET_TYPE="binary"
fi
if [ -n "${NEGATIVE_CLASS_LABEL}" ]; then
    CMD="${CMD} --negative-class-label ${NEGATIVE_CLASS_LABEL}"
    TARGET_TYPE="binary"
fi
if [ -n "${CLASS_LABELS_FILE}" ]; then
    CMD="${CMD} --class-labels-file ${CLASS_LABELS_FILE}"
    TARGET_TYPE="multiclass"
fi

CMD="${CMD} --target-type ${TARGET_TYPE}"

exec ${CMD}
