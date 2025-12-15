#!/bin/sh
# Copyright 2024 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.

# =============================================================================
# Startup script for Custom Model or MCP Server environments
# Determines which service to run based on directory contents
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# -----------------------------------------------------------------------------
# Option 1: Custom Model with DRUM Server
# Requires: custom.py file in the same directory
# -----------------------------------------------------------------------------
if [ -f "$SCRIPT_DIR/custom.py" ]; then
    echo "Starting Custom Model environment with DRUM prediction server"

    # Optional: Dump environment variables for debugging
    if [ "${ENABLE_CUSTOM_MODEL_RUNTIME_ENV_DUMP}" = 1 ]; then
        echo "Environment variables:"
        env
    fi

    # Configure UV package manager
    export UV_PROJECT=${CODE_DIR}
    unset UV_COMPILE_BYTECODE  # Disable compilation (already done in build)
    unset UV_CACHE_DIR         # Disable caching for reproducibility

    # Activate the virtual environment
    source ${VENV_PATH}/bin/activate

    # Sync dependencies using UV
    # --active: Install into the active venv instead of creating a new one
    # --frozen: Skip dependency resolution, use exact versions from lock file
    # --extra: Install the 'agentic_playground' optional dependency group
    # Note: Compilation disabled since kernel venv is already compiled
    time uv sync --frozen --active --no-progress --color never --extra agentic_playground || true

    # Start DRUM server
    echo "\nExecuting command: drum server $*\n"
    exec drum server "$@"

# -----------------------------------------------------------------------------
# Option 2: MCP Server
# Requires: app/ directory in the same location
# -----------------------------------------------------------------------------
elif [ -d "$SCRIPT_DIR/app" ]; then
    echo "Starting MCP server..."

    # Validate required environment variables
    if [ -z "$DATAROBOT_API_TOKEN" ] || [ -z "$DATAROBOT_ENDPOINT" ]; then
        echo "Error: DATAROBOT_API_TOKEN and DATAROBOT_ENDPOINT environment variables are required"
        exit 1
    fi

    # Start the MCP server
    python -m app.main

# -----------------------------------------------------------------------------
# Error: No valid entry point found
# -----------------------------------------------------------------------------
else
    echo "Error: Neither custom.py nor app/ directory found in $SCRIPT_DIR"
    echo "This script requires either:"
    echo "  - custom.py file for DRUM-based Custom Models"
    echo "  - app/ directory for MCP Server applications"
    exit 1
fi
