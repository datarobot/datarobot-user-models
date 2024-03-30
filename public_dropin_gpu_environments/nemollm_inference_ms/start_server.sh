#!/bin/sh
# Copyright 2024 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
echo "Starting Custom Model environment with NIM"
set -e

export GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "GPU count: $GPU_COUNT"

if [ "$GPU_COUNT" -eq 0 ]; then
    echo "No GPUs found in the system."
    exit 1
fi


if [ "${ENABLE_CUSTOM_MODEL_RUNTIME_ENV_DUMP}" = 1 ]; then
    echo "Environment variables:"
    env

    echo
    echo "Running NVIDIA init scripts..."
    echo
    /opt/nvidia/nvidia_entrypoint.sh /bin/true
fi


echo
echo "Starting DRUM server..."
echo
source /home/nemo/dr/bin/activate
exec drum server --gpu-predictor=nemo --logging-level=info "$@"
