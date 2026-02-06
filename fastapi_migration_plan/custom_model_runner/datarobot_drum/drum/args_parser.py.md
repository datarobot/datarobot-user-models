# Plan: custom_model_runner/datarobot_drum/drum/args_parser.py Migration

Updates to CLI argument parsing to remove Flask-specific terminology and defaults.

## Overview

The `args_parser.py` file defines the command-line arguments for DRUM. It contains several references to Flask in help messages and descriptions that should be updated to reflect the move to FastAPI.

## Required Changes

### 1. Update `--address` help message
The current help message mentions the default Flask port:
```python
# Line 336
help="Prediction server address host[:port]. Default Flask port is: 5000. The argument can also be provided by setting {} env var.".format(
    ArgumentOptionsEnvVars.ADDRESS
),
```
**Change:** Update to mention the new default port or simply remove "Flask". FastAPI/Uvicorn typically uses 8000 by default, but DRUM might still want to stick to 5000 for backward compatibility. However, the mention of "Flask" should be removed.

### 2. Update `--production` help message
The current help message describes production mode as "Flask running in multi-process":
```python
# Line 405
help=(
    "[DEPRECATED] Run prediction server in production mode, which means Flask running in multi-process. "
    f"The argument can also be provided by setting {ArgumentOptionsEnvVars.PRODUCTION} env var. "
    "(It requires --max-workers option)."
),
```
**Change:** Update to reflect that production mode now uses Uvicorn/Gunicorn with FastAPI.

## Implementation Details
- Update help strings in `CMRunnerArgsRegistry` methods: `_reg_arg_address` and `_reg_arg_production_server`.
- Ensure all other help strings are framework-agnostic.
