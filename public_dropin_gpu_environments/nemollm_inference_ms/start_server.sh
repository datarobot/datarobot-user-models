#!/bin/sh
# Copyright 2024 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
echo "Starting Custom Model environment with NIM"
set -e

if [ "${ENABLE_CUSTOM_MODEL_RUNTIME_ENV_DUMP}" = 1 ]; then
    echo "Environment variables:"
    env

    echo
    echo "Running NVIDIA init scripts..."
    echo
    /opt/nvidia/nvidia_entrypoint.sh /bin/true
fi

source /home/nemo/dr/bin/activate
echo
echo "Starting DRUM server in background..."
echo
nohup drum server --with-triton-server "$@" &

if [[ -e $CODE_DIR/custom.py ]]; then
    echo
    echo "Running custom.py..."
    echo
    python ${CODE_DIR}/custom.py
fi
deactivate

echo
echo "Starting NeMo Inference Microservice..."
echo
export MODEL_DIR="${CODE_DIR}/model-store/"
export MODEL_NAME=generic-llm

export NEMO_PORT=9998
export OPENAI_PORT=9999
export HEALTH_PORT=8081

exec nemollm_inference_ms --model $MODEL_NAME \
    --log_level=info \
    --health_port=$HEALTH_PORT \
    --openai_port=$OPENAI_PORT \
    --nemo_port=$NEMO_PORT \
    --num_gpus=$(nvidia-smi -L | wc -l)