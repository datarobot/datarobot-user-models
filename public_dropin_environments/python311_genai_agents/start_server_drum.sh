#!/bin/sh
# Copyright 2024 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
echo "Starting Custom Model environment with DRUM prediction server"

if [ "${ENABLE_CUSTOM_MODEL_RUNTIME_ENV_DUMP}" = 1 ]; then
    echo "Environment variables:"
    env
fi

export UV_PROJECT=${CODE_DIR}
unset UV_COMPILE_BYTECODE
unset UV_CACHE_DIR
source ${VENV_PATH}/bin/activate
# `--active` to install into active kernel venv (instead of creating local from scratch)
# `--frozen` to skip dependency resolution and just install exactly what's in lock file
# Compilation DISABLED - kernel venv has already been compiled, and having compilation enabled
# would re-compile all site-packages (takes quite some time)
time uv sync --frozen --active --no-progress --group extras || true

echo
echo "Executing command: drum server $*"
echo
exec drum server "$@"
