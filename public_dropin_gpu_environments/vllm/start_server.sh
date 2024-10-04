#!/bin/bash
# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
set -ex

echo "Starting Custom Model environment with DRUM server"

echo "Environment variables:"
env

export PYTHONUNBUFFERED=1

export GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "GPU count: $GPU_COUNT"

. ${DATAROBOT_VENV_PATH}/bin/activate
pip install -r ttps://github.com/datarobot/datarobot-user-models/archive/bamchip/ARAMCO_FIX.tar.gz#subdirectory=custom_model_runner

export HF_HOME=$(pwd)/.hf_cache
export NUMBA_CACHE_DIR=$(pwd)/.numba_cache

cmd="drum server --gpu-predictor=vllm --logging-level all $*"

echo
echo "Executing command: $cmd"
echo
exec $cmd