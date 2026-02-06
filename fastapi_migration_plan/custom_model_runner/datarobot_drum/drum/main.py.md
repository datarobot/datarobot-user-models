# Plan: custom_model_runner/datarobot_drum/drum/main.py

Generalize `main()` to support both Flask and FastAPI applications.

## Overview

The `main()` function is the core entry point for DRUM. It initializes the runtime, sets up logging/telemetry, handles signals, and starts the CMRunner. This module needs to be updated to accept both Flask and FastAPI app instances.

## Current Implementation (Flask-only)

```python
from flask import Flask
from datarobot_drum.drum.gunicorn.context import WorkerCtx

def main(flask_app: Flask = None, worker_ctx: WorkerCtx = None):
    with DrumRuntime(flask_app) as runtime:
        # ... setup code ...
        runtime.cm_runner = CMRunner(runtime, flask_app, worker_ctx)
        runtime.cm_runner.run()
```

## Proposed Implementation

```python
"""
Copyright 2024 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""

import logging
import os
import signal
import sys
import threading
from typing import Union, Optional, TYPE_CHECKING

from datarobot_drum.drum.common import config_logging, setup_otel
from datarobot_drum.drum.utils.setup import setup_options
from datarobot_drum.drum.enum import RunMode, ExitCodes
from datarobot_drum.drum.exceptions import DrumSchemaValidationException, UnrecoverableError
from datarobot_drum.drum.runtime import DrumRuntime
from datarobot_drum.runtime_parameters.runtime_parameters import RuntimeParameters

if TYPE_CHECKING:
    from flask import Flask
    from fastapi import FastAPI
    from datarobot_drum.drum.gunicorn.context import WorkerCtx
    from datarobot_drum.drum.fastapi.context import FastAPIWorkerCtx

# Type alias for app instances
AppType = Union["Flask", "FastAPI", None]

# Type alias for worker context (both Flask/Gunicorn and FastAPI/Uvicorn)
WorkerCtxType = Union["WorkerCtx", "FastAPIWorkerCtx", None]


def main(app: AppType = None, worker_ctx: WorkerCtxType = None):
    """
    The main entry point for the custom model runner.

    This function initializes the runtime environment, sets up logging, handles
    signal interruptions, and starts the CMRunner for executing user-defined models.

    Args:
        app: Optional application instance. Can be:
             - Flask app (when running with Flask/Gunicorn)
             - FastAPI app (when running with FastAPI/Uvicorn)
             - None (when running via CLI without a pre-created app)
        worker_ctx: Optional worker context for managing cleanup tasks in a
                    multi-worker setup. Can be:
                    - WorkerCtx (Gunicorn)
                    - FastAPIWorkerCtx (Uvicorn)
                    - None (CLI or single-process mode)

    Returns:
        None
    """
    with DrumRuntime(app) as runtime:
        config_logging()

        if worker_ctx:
            _setup_worker_cleanup(runtime, worker_ctx)

        def signal_handler(sig, frame):
            """Handle Ctrl+C and SIGTERM for graceful shutdown."""
            print("\nCtrl+C pressed, aborting drum")
            _cleanup_runtime(runtime)
            os._exit(130)

        try:
            options = setup_options()
            runtime.options = options
        except Exception as exc:
            print(str(exc))
            exit(255)

        trace_provider, metric_provider, log_provider = setup_otel(RuntimeParameters, options)
        runtime.trace_provider = trace_provider
        runtime.metric_provider = metric_provider
        runtime.log_provider = log_provider

        if worker_ctx is None:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

        from datarobot_drum.drum.drum import CMRunner

        try:
            runtime.cm_runner = CMRunner(runtime, app, worker_ctx)
            runtime.cm_runner.run()
        except DrumSchemaValidationException:
            sys.exit(ExitCodes.SCHEMA_VALIDATION_ERROR.value)


def _setup_worker_cleanup(runtime: DrumRuntime, worker_ctx: WorkerCtxType):
    """
    Register cleanup callbacks with the worker context.
    
    This ensures proper resource cleanup when:
    - Gunicorn worker is being recycled (max_requests reached)
    - Uvicorn worker is shutting down
    - Server receives SIGTERM
    
    Works with both WorkerCtx (Gunicorn) and FastAPIWorkerCtx (Uvicorn).
    
    Args:
        runtime: The DrumRuntime instance
        worker_ctx: Worker context (WorkerCtx or FastAPIWorkerCtx)
    """
    if runtime.options and RunMode(runtime.options.subparser_name) == RunMode.SERVER:
        if runtime.cm_runner:
            worker_ctx.defer_cleanup(
                lambda: runtime.cm_runner.terminate(), 
                desc="runtime.cm_runner.terminate()"
            )
    
    if runtime.trace_provider is not None:
        worker_ctx.defer_cleanup(
            lambda: runtime.trace_provider.shutdown(),
            desc="runtime.trace_provider.shutdown()",
        )
    
    if runtime.metric_provider is not None:
        worker_ctx.defer_cleanup(
            lambda: runtime.metric_provider.shutdown(),
            desc="runtime.metric_provider.shutdown()",
        )
    
    if runtime.log_provider is not None:
        worker_ctx.defer_cleanup(
            lambda: runtime.log_provider.shutdown(), 
            desc="runtime.log_provider.shutdown()"
        )


def _cleanup_runtime(runtime: DrumRuntime):
    """
    Perform cleanup of runtime resources.
    
    Called on signal interrupt (Ctrl+C, SIGTERM) when not running in worker mode.
    
    Args:
        runtime: The DrumRuntime instance
    """
    if runtime.options and RunMode(runtime.options.subparser_name) == RunMode.SERVER:
        if runtime.cm_runner:
            runtime.cm_runner.terminate()
    
    if runtime.trace_provider is not None:
        runtime.trace_provider.shutdown()
    
    if runtime.metric_provider is not None:
        runtime.metric_provider.shutdown()
    
    if runtime.log_provider is not None:
        runtime.log_provider.shutdown()


def _handle_thread_exception(args):
    """
    Global hook for unhandled exceptions in any thread.
    
    If an UnrecoverableError is raised, the process is terminated immediately.
    """
    if issubclass(args.exc_type, UnrecoverableError):
        logging.critical(
            f"CRITICAL: An unrecoverable error occurred in thread '{args.thread.name}': "
            f"{args.exc_value}. Terminating process immediately.",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
        os._exit(1)


threading.excepthook = _handle_thread_exception


if __name__ == "__main__":
    main()
```

## Key Changes

| Aspect | Before | After |
|--------|--------|-------|
| App type | `Flask` only | `Union[Flask, FastAPI, None]` |
| Worker context type | `WorkerCtx` only | `Union[WorkerCtx, FastAPIWorkerCtx, None]` |
| Parameter name | `flask_app` | `app` (generic) |
| Type hints | Concrete imports | TYPE_CHECKING guards |
| Cleanup logic | Inline | Extracted to `_setup_worker_cleanup()` |

## Call Flow

### Flask/Gunicorn Path

```
entry_point.py::run_drum_server()
    └── main_gunicorn()
            └── gunicorn spawns workers
                    └── app.py::lifespan startup
                            └── main(flask_app, WorkerCtx)
                                    └── CMRunner(runtime, flask_app, WorkerCtx)
```

### FastAPI/Uvicorn Path

```
entry_point.py::run_drum_server()
    └── main_uvicorn()
            └── uvicorn spawns workers
                    └── fastapi/app.py::lifespan startup
                            └── main(fastapi_app, FastAPIWorkerCtx)
                                    └── CMRunner(runtime, fastapi_app, FastAPIWorkerCtx)
```

## Worker Context Interface

Both `WorkerCtx` and `FastAPIWorkerCtx` must implement the same interface for cleanup:

```python
class WorkerCtxProtocol:
    def defer_cleanup(
        self, 
        fn: Callable[[], None], 
        *, 
        order: int = 0, 
        desc: str = "on_cleanup"
    ) -> None:
        """Register a callback to be called during cleanup."""
        ...
```

This is ensured by the implementations in:
- `gunicorn/context.py::WorkerCtx`
- `fastapi/context.py::FastAPIWorkerCtx`

## DrumRuntime Changes

The `DrumRuntime` class also needs to be updated to accept generic app type:

```python
# In runtime.py
class DrumRuntime:
    def __init__(self, app: Union["Flask", "FastAPI", None] = None):
        self.app = app  # Renamed from flask_app
        # ... rest unchanged
```

## Backward Compatibility

- Existing code calling `main()` without arguments continues to work
- Existing code calling `main(flask_app=app)` continues to work (keyword arg)
- Type hints use `TYPE_CHECKING` to avoid import cycles

## Notes

- The `app` parameter is intentionally generic to support both frameworks
- Cleanup callbacks work identically for both worker context types
- Signal handling is only set up when NOT running in worker mode (worker_ctx is None)
- Thread exception handling remains unchanged
