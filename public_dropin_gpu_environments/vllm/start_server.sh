#!/bin/sh
# Copyright 2024 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
echo "Starting Custom Model environment with vLLM"
set -e

export VLLM_NO_USAGE_STATS=1

echo
echo "Starting DRUM server..."
echo
source ${DATAROBOT_VENV_PATH}/bin/activate
exec drum server --gpu-predictor=vllm --logging-level=info "$@"
