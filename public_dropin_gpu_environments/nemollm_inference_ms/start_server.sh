#!/bin/sh
# Copyright 2024 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
echo "Starting Custom Model environment with NIM"
set -e

GPU_COUNT=$(nvidia-smi -L | wc -l)
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
echo "Starting NeMo Inference Microservice..."
echo
export NEMO_PORT=9998
export OPENAI_PORT=9999
export HEALTH_PORT=8081
export MODEL_NAME=generic-llm
export MODEL_DIR="${CODE_DIR}/model-store/"

nohup nemollm_inference_ms --model $MODEL_NAME \
    --log_level=info \
    --health_port=$HEALTH_PORT \
    --openai_port=$OPENAI_PORT \
    --nemo_port=$NEMO_PORT \
    --num_gpus=$GPU_COUNT &


echo
echo "Starting DRUM server..."
echo
source /home/nemo/dr/bin/activate
exec drum server --with-nemo-server "$@"
