# Plan: custom_model_runner/datarobot_drum/drum/fastapi/app.py

ASGI application entry point for FastAPI with lifespan management.

## Overview

This module defines the FastAPI application instance and handles worker lifecycle events (startup/shutdown) using FastAPI's lifespan context manager. It mirrors the functionality of `gunicorn/app.py` and `gunicorn.conf.py`.

## Proposed Implementation:

```python
"""
FastAPI ASGI application entry point.
Mirrors gunicorn/app.py functionality.
"""
import logging
import os
import shlex
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX
from datarobot_drum.drum.server import create_fastapi_app

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)

# Global worker context (equivalent to gunicorn/app.py)
_worker_ctx: Optional["FastAPIWorkerCtx"] = None


def set_worker_ctx(ctx: "FastAPIWorkerCtx"):
    """Set the global worker context."""
    global _worker_ctx
    _worker_ctx = ctx


def get_worker_ctx() -> Optional["FastAPIWorkerCtx"]:
    """Get the global worker context."""
    return _worker_ctx


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    
    Startup (equivalent to gunicorn's post_worker_init):
    - Restore sys.argv from DRUM_UVICORN_DRUM_ARGS
    - Reset MAX_WORKERS for single-worker mode
    - Create and start WorkerCtx
    
    Shutdown (equivalent to gunicorn's worker_exit):
    - Stop the worker context gracefully
    - Cleanup resources
    """
    # === STARTUP ===
    logger.info("FastAPI lifespan startup initiated")
    
    # Restore sys.argv from environment variable
    drum_args = os.environ.get("DRUM_UVICORN_DRUM_ARGS", "")
    if drum_args:
        sys.argv = shlex.split(drum_args)
        logger.debug("Restored sys.argv: %s", sys.argv)
    
    # Reset MAX_WORKERS to 1 for the worker process
    # (similar to gunicorn's post_worker_init)
    os.environ["MAX_WORKERS"] = "1"
    
    # Remove CUSTOM_MODEL_WORKERS from runtime params if set
    # This prevents nested workers
    from datarobot_drum import RuntimeParameters
    if RuntimeParameters.has("CUSTOM_MODEL_WORKERS"):
        os.environ.pop("MLOPS_RUNTIME_PARAM_CUSTOM_MODEL_WORKERS", None)
    
    # Create worker context
    from datarobot_drum.drum.fastapi.context import create_ctx
    ctx = create_ctx(app)
    set_worker_ctx(ctx)
    
    # Start the DRUM runtime (loads model, initializes predictor, etc.)
    ctx.start()
    
    logger.info("FastAPI lifespan startup complete")
    
    yield  # Application runs here
    
    # === SHUTDOWN ===
    logger.info("FastAPI lifespan shutdown initiated")
    
    ctx = get_worker_ctx()
    if ctx:
        try:
            # Graceful stop (stop background threads, etc.)
            ctx.stop()
        except Exception as e:
            logger.error("Error during context stop: %s", e)
        finally:
            # Cleanup resources (close connections, flush metrics, etc.)
            ctx.cleanup()
    
    logger.info("FastAPI lifespan shutdown complete")


def create_app() -> FastAPI:
    """
    Factory function to create the FastAPI application.
    Used when running with --factory flag or programmatic startup.
    """
    app = create_fastapi_app()
    return app


# Create the application instance with lifespan handler
# This is the entry point for Uvicorn: "app:app"
app = create_fastapi_app()

# Override the app's lifespan with our custom one
# Note: We need to recreate the app with lifespan or set it after creation
_original_lifespan = app.router.lifespan_context
app.router.lifespan_context = lifespan
```

## Alternative Implementation Using Startup/Shutdown Events

If lifespan context manager causes issues, use deprecated but simpler events:

```python
# Alternative using on_event decorators (deprecated in FastAPI 0.103+)
app = create_fastapi_app()

@app.on_event("startup")
async def startup_event():
    """Initialize DRUM runtime on worker startup."""
    logger.info("FastAPI startup event")
    
    # Restore sys.argv
    drum_args = os.environ.get("DRUM_UVICORN_DRUM_ARGS", "")
    if drum_args:
        sys.argv = shlex.split(drum_args)
    
    os.environ["MAX_WORKERS"] = "1"
    
    from datarobot_drum.drum.fastapi.context import create_ctx
    ctx = create_ctx(app)
    set_worker_ctx(ctx)
    ctx.start()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on worker shutdown."""
    logger.info("FastAPI shutdown event")
    
    ctx = get_worker_ctx()
    if ctx:
        try:
            ctx.stop()
        finally:
            ctx.cleanup()
```

## Key Differences from Gunicorn

| Aspect | Gunicorn | FastAPI/Uvicorn |
|--------|----------|-----------------|
| Startup hook | `post_worker_init(worker)` in conf.py | `lifespan` context manager or `@app.on_event("startup")` |
| Shutdown hook | `worker_exit(worker, code)` in conf.py | `lifespan` context manager or `@app.on_event("shutdown")` |
| App creation | `create_flask_app()` called in app.py | `create_fastapi_app()` with lifespan |
| Worker context | Global `_worker_ctx` variable | Same pattern |

## Notes:
- The lifespan context manager is the recommended approach for FastAPI 0.93+.
- The `@app.on_event` decorators are deprecated but may be needed for compatibility.
- Uvicorn handles SIGTERM/SIGINT signals and triggers the shutdown path automatically.
- When running with multiple workers (`workers > 1`), each worker has its own lifespan.
