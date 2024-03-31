#!/bin/sh
# Copyright 2024 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
echo "Starting Custom Model environment with Triton inference server"
set -e

if [ "${ENABLE_CUSTOM_MODEL_RUNTIME_ENV_DUMP}" = 1 ]; then
    echo "Environment variables:"
    env

    echo
    echo "Running NVIDIA init scripts..."
    echo
    /opt/nvidia/nvidia_entrypoint.sh /bin/true
fi

echo
echo "Executing command: tritonserver --model-repository=${CODE_DIR}"
echo
nohup tritonserver --model-repository=${MODEL_DIR} &

echo
echo "Executing command: drum server $*"
echo
exec drum server --gpu-predictor=triton --logging-level=info "$@"
