#!/bin/sh
# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.

# =============================================================================
# Startup script for MCP Server custom models.
#
# Copied to /opt/code/start_server.sh in the image and invoked by the platform
# for the custom-model surface only. The Workload API (code-to-workload) surface
# never runs this script -- it generates its own Dockerfile and entrypoint on top
# of this image.
#
# POSIX sh on purpose: keep it free of bashisms.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Configure UV package manager
export UV_PROJECT="${CODE_DIR:-/opt/code}"
export UV_PROJECT_ENVIRONMENT="${VENV_DIR:-/opt/venv}"
export UV_COMPILE_BYTECODE=0  # Disable compilation
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

# Create venv in code dir.
uv venv "${UV_PROJECT_ENVIRONMENT}"
# shellcheck disable=SC1091
. "${UV_PROJECT_ENVIRONMENT}/bin/activate"

# Sync dependencies using UV
# --active: Install into the active venv instead of creating a new one
# --frozen: Skip dependency resolution, use exact versions from lock file
# Note: Compilation disabled since the baked venv is already compiled
# Does not fail on errors to avoid blocking the startup of the server
uv sync --frozen --active --no-progress --color never || true

# Optional: Dump environment variables for debugging
if [ "${ENABLE_CUSTOM_MODEL_RUNTIME_ENV_DUMP}" = "1" ]; then
    echo "Environment variables:"
    env
fi

# -----------------------------------------------------------------------------
# MCP Server
# Requires: app/ directory in the same location
#
# No --root_path / ROOT_PATH_ARG is threaded through here, unlike the dragent
# branch in python311_genai_agents. A deployed server is served under
# https://<endpoint>/deployments/<id>/directAccess/, and drmcp already applies
# that prefix itself: DRMCPConfig reads URL_PREFIX straight from the environment
# as `mount_path` (datarobot_genai/drmcp/core/config.py) and every route is
# registered through prefix_mount_path(). Passing the prefix again would
# double-prefix it.
# -----------------------------------------------------------------------------
if [ -d "$SCRIPT_DIR/app" ]; then
    echo "Starting Custom Model environment with MCP server"

    # Set Python path to script directory for module imports
    export PYTHONPATH="$SCRIPT_DIR"

    # Start the MCP server
    exec python -m app.main
fi

# -----------------------------------------------------------------------------
# Error: No valid entry point found
# -----------------------------------------------------------------------------
echo "Error: No valid entry point found in $SCRIPT_DIR"
echo "This environment requires an app/ directory containing an MCP server"
echo "exposing a runnable app.main module."
exit 1
