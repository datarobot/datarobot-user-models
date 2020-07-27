#!/bin/sh
cd /opt/code/ || exit 1
export PYTHONPATH=/opt/code


H2O_POJO=$(ls /opt/model | grep .java)
if [ $H2O_POJO == "" ]; then
    echo nothing there
else
    javac -cp /opt/code_dep/h2o.jar /opt/model/$H2O_POJO
fi


CMD="drum server -cd . --address 0.0.0.0:8080 --with-error-server"

if [ ! -z "${POSITIVE_CLASS_LABEL}" ]; then
    CMD="${CMD} --positive-class-label ${POSITIVE_CLASS_LABEL}"
fi
if [ ! -z "${NEGATIVE_CLASS_LABEL}" ]; then
    CMD="${CMD} --negative-class-label ${NEGATIVE_CLASS_LABEL}"
fi

exec ${CMD}