#!/bin/sh
cd /opt/code/ || exit 1
export PYTHONPATH=/opt/code

CMD="drum server -cd . --target-type ${TARGET_TYPE} --address 0.0.0.0:8080"

# Uncomment the following line to switch from Flask to uwsgi server
# WITH_UWSGI="1"
if [ -n "${WITH_UWSGI}" ]; then
  CMD="${CMD} --production --max-workers 1 --show-stacktrace"
else
  CMD="${CMD} --with-error-server"
fi

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
