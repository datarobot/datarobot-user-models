#!/bin/sh
# Copyright 2024 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
echo "Starting Custom Model environment with Triton inference server"

if [ "${ENABLE_CUSTOM_MODEL_RUNTIME_ENV_DUMP}" = 1 ]; then
    echo "Environment variables:"
    env
fi

export MODEL_DIR="${CODE_DIR}/model_repository/"

echo
echo "Executing command: tritonserver --model-repository=${MODEL_DIR}"
echo
exec nohup tritonserver --model-repository=${MODEL_DIR} > log.txt 2>&1 &

echo
echo "Executing command: drum server $*"
echo
exec drum server "$@"
