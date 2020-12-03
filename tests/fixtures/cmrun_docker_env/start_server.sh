#!/bin/sh
cd /opt/code/ || exit 1
export PYTHONPATH=$PYTHONPATH:/opt/code

CMD="drum -cd . --server 0.0.0.0:8080"

if [ -n "${POSITIVE_CLASS_LABEL}" ]; then
    CMD="${CMD} --positive-class-label ${POSITIVE_CLASS_LABEL}"
fi
if [ -n "${NEGATIVE_CLASS_LABEL}" ]; then
    CMD="${CMD} --negative-class-label ${NEGATIVE_CLASS_LABEL}"
fi

${CMD}
