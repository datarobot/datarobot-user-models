# Plan: custom_model_runner/datarobot_drum/drum/fastapi/error_server.py

Implementation of the FastAPI-based Error Server.

## Overview

The Error Server is a fallback server that starts when the DRUM model fails to initialize. It keeps the container alive and reports the initialization error on all endpoints. This is critical for infrastructure (like Kubernetes or SageMaker) that expects the container to be "ready" and responsive.

## Proposed Implementation

This logic will be primarily located in `runtime.py` or a dedicated `error_server.py` module.

```python
"""
FastAPI Error Server implementation.
"""
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

def run_error_server_fastapi(host: str, port: int, exc_value: Exception):
    """
    Start a minimal FastAPI server that reports initialization errors.
    
    - Returns 200 OK for /ping, /health to keep infrastructure happy.
    - Returns 513 (DRUM Pipeline Error) for all other endpoints.
    """
    app = FastAPI(title="DRUM Error Server (FastAPI)")
    
    error_msg = {
        "message": f"ERROR: Model initialization failed: {exc_value}",
        "status": "error"
    }
    
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def catch_all(request: Request, path: str):
        # Normalize path
        clean_path = path.rstrip("/")
        
        # Infrastructure probes should succeed
        if clean_path in ["ping", "health", ""]:
            return JSONResponse(
                content={
                    "message": "Error Server Running",
                    "error": str(exc_value)
                },
                status_code=200
            )
            
        # All prediction/info endpoints return the initialization error
        return JSONResponse(
            content=error_msg,
            status_code=513 # HTTP_513_DRUM_PIPELINE_ERROR
        )
        
    logger.info("Starting FastAPI Error Server on %s:%s", host, port)
    
    # Run uvicorn with minimal workers and logging
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        workers=1
    )
```

## Integration with `DrumRuntime`

The `DrumRuntime.__exit__` method should detect the failure and start the appropriate error server:

```python
# In runtime.py
def __exit__(self, exc_type, exc_val, exc_tb):
    if exc_type and getattr(self.options, "with_error_server", False):
        server_type = self._get_requested_server_type()
        
        if server_type == "fastapi":
            from datarobot_drum.drum.fastapi.error_server import run_error_server_fastapi
            run_error_server_fastapi(host, port, exc_val)
        else:
            run_error_server(host, port, exc_val, self.app)
```

## Key Differences from Flask Error Server

| Aspect | Flask Error Server | FastAPI Error Server |
|--------|--------------------|----------------------|
| Framework | Flask | FastAPI |
| Server | `app.run()` (Werkzeug) | `uvicorn.run()` |
| Catch-all | `@app.route('/<path:path>')` | `@app.api_route("/{path:path}")` |
| Status Code | 513 | 513 |

## Notes:
- The error server must be extremely lightweight and have zero dependencies on the user's model code.
- It must respond to all methods (GET, POST, etc.) for catch-all routes.
- The 513 status code is standard for DRUM to indicate a pipeline error.
