#!/bin/sh
cd /opt/code/ || exit 1
export PYTHONPATH=/opt/code

CMD="drum server -cd . --address 0.0.0.0:8080 --with-error-server --production --max-workers 1"

if [ ! -z "${POSITIVE_CLASS_LABEL}" ]; then
    CMD="${CMD} --positive-class-label ${POSITIVE_CLASS_LABEL}"
fi
if [ ! -z "${NEGATIVE_CLASS_LABEL}" ]; then
    CMD="${CMD} --negative-class-label ${NEGATIVE_CLASS_LABEL}"
fi

exec ${CMD}
