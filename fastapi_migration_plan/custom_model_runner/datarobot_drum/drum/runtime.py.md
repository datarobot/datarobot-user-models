# Plan: custom_model_runner/datarobot_drum/drum/runtime.py

Generalize `DrumRuntime` to support both Flask and FastAPI.

## Overview

The `DrumRuntime` class manages the lifecycle of the DRUM environment, including options, providers, and the `CMRunner`. It needs to be updated to accept a generic web application instance.

## Proposed Implementation:

### 1. Update `__init__` signature

```python
from typing import Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from flask import Flask
    from fastapi import FastAPI

class DrumRuntime:
    def __init__(self, app: Union["Flask", "FastAPI", None] = None):
        """
        Initialize the DrumRuntime.
        
        Args:
            app: Optional web application instance (Flask or FastAPI).
                 Renamed from flask_app for framework-agnostic support.
        """
        self.app = app  # Renamed from flask_app
        self.cm_runner = None
        self.options = None
        self.trace_provider = None
        self.metric_provider = None
        self.log_provider = None
        # ...
```

### 2. Update `_setup_stdout_flusher`

The `StdoutFlusher` is currently initialized and managed within the server lifecycle. Ensure it works correctly with the new `app` attribute.

```python
def _setup_stdout_flusher(self):
    """
    Initialize StdoutFlusher if running in server mode.
    """
    if self.app:
        # The flusher activity tracking should be integrated with
        # framework-specific middleware or event handlers.
        pass
```

### 3. Lifecycle Management in `__enter__` and `__exit__`

The `DrumRuntime` acts as a context manager. Ensure that entering and exiting the context correctly handles resource initialization and cleanup for both frameworks.

```python
def __enter__(self):
    return self

def __exit__(self, exc_type, exc_val, exc_tb):
    """
    Ensure all providers and the runner are shut down on exit.
    If initialization failed and with_error_server is set, start the error server.
    """
    if self.cm_runner:
        self.cm_runner.terminate()
    
    if self.trace_provider:
        self.trace_provider.shutdown()
    
    if self.metric_provider:
        self.metric_provider.shutdown()
        
    if self.log_provider:
        self.log_provider.shutdown()

    if not exc_type:
        return True

    # Error server logic
    if self.options and getattr(self.options, "with_error_server", False) and not self.initialization_succeeded:
        host_port_list = self.options.address.split(":", 1)
        host = host_port_list[0]
        port = int(host_port_list[1]) if len(host_port_list) == 2 else None
        
        server_type = "flask"
        if hasattr(self.options, "server_type"):
            server_type = self.options.server_type
        elif os.environ.get("MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE") in ["fastapi", "uvicorn"]:
            server_type = "fastapi"

        if server_type == "fastapi":
            run_error_server_fastapi(host, port, exc_val)
        else:
            run_error_server(host, port, exc_val, self.app)

    return False


def run_error_server_fastapi(host: str, port: int, exc_value: Exception):
    """
    FastAPI version of the error server.
    Started when the model fails to load but we want to keep the container alive 
    and reporting errors on all endpoints.
    
    This ensures that Kubernetes/Sagemaker see the container as 'ready' (if configured)
    but any actual request will return the initialization error.
    """
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    
    app = FastAPI(title="DRUM Error Server (FastAPI)")
    
    error_msg = {"message": f"ERROR: {exc_value}"}
    status_code = 513  # HTTP_513_DRUM_PIPELINE_ERROR
    
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def catch_all(request: Request, path: str):
        # We still want /ping and /health to return 200 so the container stays up
        if path.rstrip("/") in ["ping", "health", ""]:
            return JSONResponse(content={"message": "Error Server Running", "error": str(exc_value)}, status_code=200)
            
        return JSONResponse(content=error_msg, status_code=status_code)
        
    logger.info("Starting FastAPI Error Server on %s:%s", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
```

## Migration Notes:

| Before | After | Reason |
|--------|-------|--------|
| `self.flask_app` | `self.app` | Support both Flask and FastAPI |
| `DrumRuntime(flask_app=app)` | `DrumRuntime(app=app)` | Framework-agnostic initialization |

## Notes:

- The `app` attribute is passed to `CMRunner` during its initialization in `main()`.
- Resource providers (OTEL) are managed by `DrumRuntime` and correctly shut down regardless of the web framework used.
