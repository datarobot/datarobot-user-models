#!/bin/sh
# Copyright 2024 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
echo "Starting Custom Model environment with vLLM"
set -e

export VLLM_CONFIGURE_LOGGING=0
export VLLM_NO_USAGE_STATS=1

# TODO: this is nvidia specific but vLLM supports other GPUs/TPUs/CPUs
export GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "GPU count: $GPU_COUNT"

if [ "$GPU_COUNT" -eq 0 ]; then
    echo "No GPUs found in the system."
    exit 1
fi

echo
echo "Starting DRUM server..."
echo
source ${DATAROBOT_VENV_PATH}/bin/activate
exec drum server --gpu-predictor=vllm --logging-level=info "$@"