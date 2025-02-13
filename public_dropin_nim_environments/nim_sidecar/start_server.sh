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
fi

echo
echo "Starting DRUM server..."
echo
exec drum server --sidecar --gpu-predictor=nim --logging-level=info "$@"
