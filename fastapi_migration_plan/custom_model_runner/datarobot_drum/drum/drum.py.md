# Plan: custom_model_runner/datarobot_drum/drum/drum.py

Update `CMRunner` to support both Flask and FastAPI app instances.

## Overview

The `CMRunner` class orchestrates the DRUM runtime. It receives an optional app instance and worker context, and based on the run mode, executes the appropriate pipeline (score, server, fit, etc.). For SERVER mode, it needs to pass the app to `PredictionServer`.

## Current Implementation (Flask-only)

```python
class CMRunner:
    def __init__(self, runtime, flask_app=None, worker_ctx=None):
        self.runtime = runtime
        self.flask_app = flask_app  # Flask app object
        self.worker_ctx = worker_ctx  # Gunicorn WorkerCtx
        # ...
```

## Proposed Implementation

### 1. Update `__init__` signature

```python
from typing import Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from flask import Flask
    from fastapi import FastAPI
    from datarobot_drum.drum.gunicorn.context import WorkerCtx
    from datarobot_drum.drum.fastapi.context import FastAPIWorkerCtx

# Type aliases
AppType = Union["Flask", "FastAPI", None]
WorkerCtxType = Union["WorkerCtx", "FastAPIWorkerCtx", None]


class CMRunner:
    def __init__(
        self, 
        runtime, 
        app: AppType = None, 
        worker_ctx: WorkerCtxType = None
    ):
        """
        Initialize the Custom Model Runner.
        
        Args:
            runtime: The DrumRuntime instance containing options and state
            app: Optional web application instance. Can be:
                 - Flask app (when running with Flask/Gunicorn)
                 - FastAPI app (when running with FastAPI/Uvicorn)
                 - None (when running via CLI without a pre-created app)
            worker_ctx: Optional worker context for cleanup management. Can be:
                        - WorkerCtx (Gunicorn)
                        - FastAPIWorkerCtx (Uvicorn)
                        - None (CLI or single-process mode)
        """
        self.runtime = runtime
        self.app = app  # Generic app (Flask or FastAPI)
        self.worker_ctx = worker_ctx  # Worker context (WorkerCtx or FastAPIWorkerCtx)
        self.options = runtime.options
        self.options.model_config = read_model_metadata_yaml(self.options.code_dir)
        # ... rest of initialization unchanged
```

### 2. Update `_run_predictions()` method

The `_run_predictions()` method creates a `PredictionServer` and runs it. It needs to pass the generic app:

```python
def _run_predictions(self):
    """
    Run the prediction server pipeline.
    
    Creates a PredictionServer instance and materializes it with the appropriate
    web framework (Flask or FastAPI).
    """
    from datarobot_drum.drum.root_predictors.prediction_server import PredictionServer
    
    # Build params dict with model configuration
    params = self._prepare_prediction_params()
    
    # Create PredictionServer with the generic app
    # PredictionServer.materialize() will detect the app type and create
    # appropriate routes (Blueprint for Flask, APIRouter for FastAPI)
    prediction_server = PredictionServer(params, app=self.app)
    
    # Start watchdog if enabled (NIM sidecar monitoring)
    # Note: _start_watchdog is called inside PredictionServer.materialize() 
    # when worker_ctx is available. See context.py.md for details.
    
    # Materialize the server (create routes, load extensions)
    return prediction_server.materialize()
```

### 3. Add helper method to detect app type

```python
def _is_fastapi_app(self) -> bool:
    """
    Check if the current app is a FastAPI instance.
    
    Returns:
        True if self.app is a FastAPI instance, False otherwise.
    """
    if self.app is None:
        return False
    
    # Check by class name to avoid importing FastAPI
    return type(self.app).__name__ == "FastAPI"


def _is_flask_app(self) -> bool:
    """
    Check if the current app is a Flask instance.
    
    Returns:
        True if self.app is a Flask instance, False otherwise.
    """
    if self.app is None:
        return False
    
    # Check by class name to avoid importing Flask
    return type(self.app).__name__ == "Flask"


def _get_server_type(self) -> str:
    """
    Get the server type string for the current app.
    
    Returns:
        "fastapi", "flask", or "none" based on self.app type.
    """
    if self._is_fastapi_app():
        return "fastapi"
    elif self._is_flask_app():
        return "flask"
    else:
        return "none"
```

### 4. Update terminate() for both frameworks

```python
def terminate(self):
    """
    Terminate the CMRunner and cleanup resources.
    
    Works for both Flask and FastAPI servers.
    """
    self.logger.info("Terminating CMRunner...")
    
    # Stop any background tasks
    if hasattr(self, '_background_tasks'):
        for task in self._background_tasks:
            task.cancel()
    
    # Close any open resources
    if hasattr(self, '_predictor') and self._predictor is not None:
        if hasattr(self._predictor, 'cleanup'):
            self._predictor.cleanup()
    
    self.logger.info("CMRunner terminated")
```

## Complete Class Signature

```python
class CMRunner:
    """
    Custom Model Runner - orchestrates DRUM execution.
    
    Supports running models in various modes:
    - SCORE: Batch prediction from file
    - SERVER: HTTP prediction server (Flask or FastAPI)
    - FIT: Model training
    - PERF_TEST: Performance testing
    - VALIDATION: Model validation
    - NEW: Generate model template
    - PUSH: Push model to DataRobot
    
    Attributes:
        runtime: DrumRuntime instance
        app: Web application instance (Flask or FastAPI) or None
        worker_ctx: Worker context (WorkerCtx or FastAPIWorkerCtx) or None
        options: Parsed command-line options
        logger: Logger instance
        run_mode: Current execution mode (RunMode enum)
        target_type: Model target type (TargetType enum)
    """
    
    def __init__(
        self, 
        runtime, 
        app: AppType = None, 
        worker_ctx: WorkerCtxType = None
    ):
        ...
    
    def run(self):
        """Execute the appropriate pipeline based on run_mode."""
        ...
    
    def terminate(self):
        """Gracefully terminate and cleanup resources."""
        ...
    
    # Private methods
    def _run_predictions(self):
        """Run prediction server pipeline."""
        ...
    
    def _is_fastapi_app(self) -> bool:
        """Check if app is FastAPI."""
        ...
    
    def _is_flask_app(self) -> bool:
        """Check if app is Flask."""
        ...
    
    def _get_server_type(self) -> str:
        """Get server type string."""
        ...
```

## Migration Notes

### Renamed Attributes

| Before | After | Reason |
|--------|-------|--------|
| `self.flask_app` | `self.app` | Generic for both frameworks |

### Parameter Changes

| Method | Before | After |
|--------|--------|-------|
| `__init__` | `flask_app=None` | `app=None` |
| `_run_predictions` | Uses `self.flask_app` | Uses `self.app` |

### Type Annotations

Using `TYPE_CHECKING` guard to avoid circular imports:

```python
if TYPE_CHECKING:
    from flask import Flask
    from fastapi import FastAPI
```

## Backward Compatibility

- Keyword argument `flask_app` can be kept as deprecated alias:

```python
def __init__(
    self, 
    runtime, 
    app: AppType = None, 
    worker_ctx: WorkerCtxType = None,
    flask_app: AppType = None,  # Deprecated, for backward compatibility
):
    # Handle deprecated parameter
    if flask_app is not None and app is None:
        import warnings
        warnings.warn(
            "flask_app parameter is deprecated, use app instead",
            DeprecationWarning,
            stacklevel=2
        )
        app = flask_app
    
    self.app = app
    # ...
```

## Notes

- The app instance is passed through from `main()` → `CMRunner` → `PredictionServer`
- `PredictionServer.materialize()` handles the framework-specific route creation
- Worker context is used for cleanup registration regardless of framework
- Type checking uses string annotations and TYPE_CHECKING to avoid import cycles
