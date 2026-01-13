# Plan: custom_model_runner/datarobot_drum/drum/server.py

Implement FastAPI application creation with middleware equivalent to Flask's before_request/after_request.

## Overview

Add FastAPI support alongside existing Flask functions. After migration is complete, Flask functions will be removed.

## Changes:

### 1. New imports
```python
import datetime
from typing import Optional, Union
import uuid

from fastapi import FastAPI, Request, Response
from fastapi.routing import APIRouter
from starlette.middleware.base import BaseHTTPMiddleware

# Keep existing Flask imports during transition
import flask
from flask import Flask, Blueprint, request
```

### 2. Create FastAPI middleware classes

```python
import asyncio
from datarobot_drum import RuntimeParameters


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle X-Request-ID header.
    Equivalent to Flask's before_request/after_request for request ID.
    """
    async def dispatch(self, request: Request, call_next):
        # Get or generate request ID
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Store in request state
        request.state.request_id = request_id
        request.state.request_start_time = datetime.datetime.now()
        
        # Set context variable
        token = ctx_request_id.set(request_id)
        
        try:
            response = await call_next(request)
        finally:
            ctx_request_id.reset(token)
        
        # Add response headers
        response.headers[HEADER_REQUEST_ID] = request_id
        response.headers[HEADER_DRUM_VERSION] = drum_version
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log requests with status >= 400.
    Equivalent to Flask's after_request logging.
    """
    async def dispatch(self, request: Request, call_next):
        start_time = datetime.datetime.now()
        
        response = await call_next(request)
        
        if response.status_code >= 400:
            request_string = f"{request.method} {request.url.path}"
            total_time = datetime.datetime.now() - start_time
            request_time = total_time.total_seconds()
            request_id = getattr(request.state, 'request_id', 'unknown')
            
            logger.info(
                "API [%s] request time: %s sec, request_id: %s",
                request_string,
                request_time,
                request_id,
            )
        
        return response


class StdoutFlusherMiddleware(BaseHTTPMiddleware):
    """
    Middleware to update last activity time for StdoutFlusher.
    Replaces Flask's after_request logic for stdout flushing.
    """
    def __init__(self, app, flusher=None):
        super().__init__(app)
        self.flusher = flusher

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if self.flusher:
            # Update activity time after request is processed
            self.flusher.set_last_activity_time()
        return response


class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce request timeout.
    Equivalent to Flask's TimeoutWSGIRequestHandler.
    
    Note: Uvicorn's timeout_keep_alive is for connection keep-alive, not request timeout.
    This middleware provides actual request-level timeout enforcement.
    """
    
    DEFAULT_TIMEOUT = 3600  # seconds
    
    def __init__(self, app, timeout: int = None):
        super().__init__(app)
        if timeout is not None:
            self.timeout = timeout
        elif RuntimeParameters.has("DRUM_CLIENT_REQUEST_TIMEOUT"):
            self.timeout = int(RuntimeParameters.get("DRUM_CLIENT_REQUEST_TIMEOUT"))
        else:
            self.timeout = self.DEFAULT_TIMEOUT
    
    async def dispatch(self, request: Request, call_next):
        # Skip timeout for health check endpoints
        if request.url.path.rstrip("/") in ["/ping", "/", "/health"]:
            return await call_next(request)
        
        if self.timeout <= 0:
            # Timeout disabled
            return await call_next(request)
        
        try:
            # Note: asyncio.wait_for will cancel the call_next(request) coroutine on timeout.
            # If the coroutine is awaiting loop.run_in_executor(), the future will be cancelled,
            # but the actual thread in the executor will continue to run until completion
            # as Python threads cannot be forcefully terminated.
            return await asyncio.wait_for(
                call_next(request),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            logger.error(
                "Request timeout after %s seconds: %s %s. "
                "Note: Background prediction thread may still be running.",
                self.timeout,
                request.method,
                request.url.path
            )
            from fastapi.responses import JSONResponse
            return JSONResponse(
                content={
                    "message": f"ERROR: Request timeout after {self.timeout} seconds"
                },
                status_code=504  # Gateway Timeout
            )


class ContentLengthLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to limit the size of the request body.
    """
    def __init__(self, app, max_content_length: int):
        super().__init__(app)
        self.max_content_length = max_content_length

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_content_length:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                content={"message": f"Request body too large. Limit is {self.max_content_length} bytes"},
                status_code=413
            )
        return await call_next(request)
```

### 2a. CORS Middleware (Optional)

If CORS support is needed (e.g., for browser-based clients), add CORS middleware:

```python
from fastapi.middleware.cors import CORSMiddleware

def add_cors_middleware(app: FastAPI):
    """
    Add CORS middleware if enabled via runtime parameters.
    
    Runtime Parameters:
    - DRUM_CORS_ENABLED: "true" to enable CORS
    - DRUM_CORS_ORIGINS: Comma-separated list of allowed origins (default: "*")
    - DRUM_CORS_METHODS: Comma-separated list of allowed methods (default: "GET,POST,PUT,DELETE,OPTIONS")
    - DRUM_CORS_HEADERS: Comma-separated list of allowed headers (default: "*")
    """
    if not RuntimeParameters.has("DRUM_CORS_ENABLED"):
        return
    
    if str(RuntimeParameters.get("DRUM_CORS_ENABLED")).lower() not in ["true", "1", "yes"]:
        return
    
    # Get configuration
    origins = ["*"]
    if RuntimeParameters.has("DRUM_CORS_ORIGINS"):
        origins_str = str(RuntimeParameters.get("DRUM_CORS_ORIGINS"))
        origins = [o.strip() for o in origins_str.split(",")]
    
    methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    if RuntimeParameters.has("DRUM_CORS_METHODS"):
        methods_str = str(RuntimeParameters.get("DRUM_CORS_METHODS"))
        methods = [m.strip() for m in methods_str.split(",")]
    
    headers = ["*"]
    if RuntimeParameters.has("DRUM_CORS_HEADERS"):
        headers_str = str(RuntimeParameters.get("DRUM_CORS_HEADERS"))
        headers = [h.strip() for h in headers_str.split(",")]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=methods,
        allow_headers=headers,
    )
    
    logger.info("CORS middleware enabled with origins: %s", origins)
```

### 3. Create FastAPI app registration function

```python
def get_fastapi_app(api_router: APIRouter, app: Optional[FastAPI] = None) -> FastAPI:
    """
    Register an API router with the FastAPI app.
    Equivalent to get_flask_app().
    
    Args:
        api_router: The APIRouter containing endpoint definitions
        app: Optional existing FastAPI app instance
        
    Returns:
        Configured FastAPI application
    """
    if app is None:
        app = create_fastapi_app()
    
    url_prefix = os.environ.get(URL_PREFIX_ENV_VAR_NAME, "")
    app.include_router(api_router, prefix=url_prefix)
    
    return app
```

### 4. Add FastAPI exception handlers

```python
def setup_fastapi_exception_handlers(app: FastAPI):
    """
    Configure global exception handlers for FastAPI.
    """
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.exception(exc)
        
        if isinstance(exc, HTTPException):
            return JSONResponse(
                content={"error": str(exc.detail)},
                status_code=exc.status_code
            )
        
        return JSONResponse(
            content={"message": f"ERROR: {exc}"},
            status_code=HTTP_500_INTERNAL_SERVER_ERROR
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            content={"error": str(exc.detail)},
            status_code=exc.status_code
        )
```

### 5. Create FastAPI app factory

```python
def create_fastapi_app() -> FastAPI:
    """
    Create and configure a FastAPI application with all necessary middleware.
    
    CRITICAL: Middleware Execution Order
    FastAPI (Starlette) executes middleware in REVERSE order of addition for the 
    request phase, and in the SAME order for the response phase (wrapping).
    
    Order of addition here:
    1. RequestLoggingMiddleware (Outer)
    2. RequestTimeoutMiddleware
    3. ContentLengthLimitMiddleware
    4. RequestIDMiddleware (Inner)
    
    Actual Execution Order (Request):
    RequestID -> ContentLength -> Timeout -> Logging -> [App Router]
    
    Actual Execution Order (Response):
    [App Router] -> RequestID -> ContentLength -> Timeout -> Logging
    """
    from datarobot_drum import RuntimeParameters
    from datarobot_drum.drum.description import version as drum_version
    
    docs_url = None
    if str(RuntimeParameters.get("DRUM_FASTAPI_ENABLE_DOCS")).lower() in ["true", "1", "yes"]:
        docs_url = "/docs"

    fastapi_app = FastAPI(
        title="DRUM Prediction Server",
        version=drum_version,
        docs_url=docs_url,
        redoc_url=None,
    )
    
    # 1. Logging (Outer)
    fastapi_app.add_middleware(RequestLoggingMiddleware)
    
    # 2. Timeout
    fastapi_app.add_middleware(RequestTimeoutMiddleware)
    
    # 3. Content Length
    max_upload_size = 100 * 1024 * 1024  # Default 100MB
    if RuntimeParameters.has("DRUM_FASTAPI_MAX_UPLOAD_SIZE"):
        max_upload_size = int(RuntimeParameters.get("DRUM_FASTAPI_MAX_UPLOAD_SIZE"))
    fastapi_app.add_middleware(ContentLengthLimitMiddleware, max_content_length=max_upload_size)

    # 4. Request ID (Inner)
    fastapi_app.add_middleware(RequestIDMiddleware)
    
    # Optional: CORS (would be outermost if added last)
    add_cors_middleware(fastapi_app)
    
    # Setup exception handlers
    setup_fastapi_exception_handlers(fastapi_app)
    
    return fastapi_app
```


def create_fastapi_app_dev() -> FastAPI:
    """
    Create FastAPI application for development with Swagger UI enabled.
    """
    from datarobot_drum.drum.description import version as drum_version
    
    fastapi_app = FastAPI(
        title="DRUM Prediction Server",
        version=drum_version,
        docs_url="/docs",    # Enable Swagger UI
        redoc_url="/redoc",  # Enable ReDoc
    )
    
    fastapi_app.add_middleware(RequestLoggingMiddleware)
    fastapi_app.add_middleware(RequestTimeoutMiddleware)
    fastapi_app.add_middleware(RequestIDMiddleware)
    add_cors_middleware(fastapi_app)
    setup_fastapi_exception_handlers(fastapi_app)
    
    return fastapi_app
```

## Preserved constants (used by both Flask and FastAPI)

```python
HTTP_200_OK = 200
HTTP_400_BAD_REQUEST = 400
HTTP_404_NOT_FOUND = 404
HTTP_422_UNPROCESSABLE_ENTITY = 422
HTTP_500_INTERNAL_SERVER_ERROR = 500
HTTP_513_DRUM_PIPELINE_ERROR = 513

HEADER_REQUEST_ID = "X_Request_ID"
HEADER_DRUM_VERSION = "X-Drum-Version"
```

## Middleware Comparison Table

| Feature | Flask | FastAPI |
|---------|-------|---------|
| Request ID | `before_request()` / `after_request()` | `RequestIDMiddleware` |
| Request Logging | `after_request()` | `RequestLoggingMiddleware` |
| Request Timeout | `TimeoutWSGIRequestHandler` | `RequestTimeoutMiddleware` |
| CORS | Flask-CORS extension | `CORSMiddleware` |
| Exception Handling | `@app.errorhandler()` | `@app.exception_handler()` |

## Runtime Parameters for Middleware

| Parameter | Description | Default |
|-----------|-------------|---------|
| `DRUM_CLIENT_REQUEST_TIMEOUT` | Request timeout in seconds | 3600 |
| `DRUM_FASTAPI_MAX_UPLOAD_SIZE` | Maximum request body size in bytes | 104857600 |
| `DRUM_CORS_ENABLED` | Enable CORS middleware | false |
| `DRUM_CORS_ORIGINS` | Allowed CORS origins | "*" |
| `DRUM_CORS_METHODS` | Allowed CORS methods | "GET,POST,PUT,DELETE,OPTIONS" |
| `DRUM_CORS_HEADERS` | Allowed CORS headers | "*" |

## Notes:
- The existing Flask functions (`create_flask_app`, `get_flask_app`, `base_api_blueprint`, `before_request`, `after_request`) remain unchanged during the transition.
- After Flask removal, only FastAPI functions will remain.
- Middleware order matters: last added = first executed.
- Request timeout middleware skips health check endpoints (`/ping`, `/`, `/health`) to allow Kubernetes probes during long-running requests.
- CORS middleware is optional and only enabled via `DRUM_CORS_ENABLED` runtime parameter.
- `StdoutFlusherMiddleware` is added in `PredictionServer._materialize_fastapi()` to have access to the flusher instance.