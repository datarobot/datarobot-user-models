#!/bin/sh
cd /opt/code/ || exit 1
export PYTHONPATH=/opt/code

CMD="drum server -cd . --target-type ${TARGET_TYPE} --address 0.0.0.0:8080 --production --max-workers 1 --show-stacktrace"

if [ -n "${POSITIVE_CLASS_LABEL}" ]; then
    CMD="${CMD} --positive-class-label ${POSITIVE_CLASS_LABEL}"
fi
if [ -n "${NEGATIVE_CLASS_LABEL}" ]; then
    CMD="${CMD} --negative-class-label ${NEGATIVE_CLASS_LABEL}"
fi
if [ -n "${CLASS_LABELS_FILE}" ]; then
    CMD="${CMD} --class-labels-file ${CLASS_LABELS_FILE}"
fi

echo "${CMD}"
exec ${CMD}
