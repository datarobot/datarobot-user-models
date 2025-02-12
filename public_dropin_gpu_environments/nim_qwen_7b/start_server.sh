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

if [ "$GPU_COUNT" -gt 1 ]; then
    echo "Multiple GPUs found; at least 16G of shared memory is recommened for non-NVLink GPUs."
    df -h /dev/shm
fi

echo ""
echo "Availble NIM Profiles:"
list-model-profiles

if [ "${ENABLE_CUSTOM_MODEL_RUNTIME_ENV_DUMP}" = 1 ]; then
    echo "Environment variables:"
    env
fi


echo
echo "Starting DRUM server..."
echo
. ${DATAROBOT_VENV_PATH}/bin/activate
exec drum server --gpu-predictor=nim --logging-level=info "$@"
