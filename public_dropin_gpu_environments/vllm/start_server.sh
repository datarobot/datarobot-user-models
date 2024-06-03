#!/bin/sh
# Copyright 2024 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
echo "Starting Custom Model environment with vLLM"
set -e

export GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "GPU count: $GPU_COUNT"

# TODO: enable uwsgi with multiple workers after we are sure we only spin up
#   one instance of vLLM and the load_model hook is only executed once.
#export PRODUCTION=1
#export MAX_WORKERS=3

echo
echo "Starting DRUM server..."
echo
. ${DATAROBOT_VENV_PATH}/bin/activate
exec drum server --gpu-predictor=vllm --logging-level=info "$@"
