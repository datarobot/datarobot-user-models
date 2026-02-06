# Plan: custom_model_runner/datarobot_drum/drum/root_predictors/prediction_server.py

Refactor `PredictionServer` to support FastAPI routing alongside Flask, with memory-efficient request handling.

## Overview

The `PredictionServer` class will be updated to work with both Flask and FastAPI apps. Key improvements:
- Centralized prediction logic callable from both frameworks
- Memory-efficient streaming for large payloads
- Proper resource cleanup via worker context

## Changes:

### 1. Imports

```python
# Add new imports
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Union, Optional, Callable, AsyncIterator, Dict, Any
from tempfile import SpooledTemporaryFile
from functools import partial

from fastapi import FastAPI, APIRouter, Request, Response, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
import httpx  # Async HTTP client for proxy endpoints

# Keep existing Flask imports for backward compatibility
from flask import Flask, Blueprint, Response as FlaskResponse, jsonify, request as flask_request

from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)
```

### 2. Memory-Efficient File Handling

```python
# Constants for memory management
SPOOL_MAX_SIZE = 10 * 1024 * 1024  # 10MB - files smaller than this stay in memory
CHUNK_SIZE = 64 * 1024  # 64KB chunks for streaming


class SpooledUploadFile:
    """
    Memory-efficient wrapper for uploaded files.
    
    Small files (< SPOOL_MAX_SIZE) stay in memory.
    Large files are spooled to disk automatically.
    """
    
    def __init__(self, upload_file: UploadFile, max_memory: int = SPOOL_MAX_SIZE):
        self._upload_file = upload_file
        self._spooled = SpooledTemporaryFile(max_size=max_memory, mode='w+b')
        self._size = 0
        self._fully_read = False
        self.filename = upload_file.filename
        self.content_type = upload_file.content_type
    
    async def read_to_spool(self) -> int:
        """Read entire file into spool, return size."""
        if self._fully_read:
            return self._size
        
        while True:
            chunk = await self._upload_file.read(CHUNK_SIZE)
            if not chunk:
                break
            self._spooled.write(chunk)
            self._size += len(chunk)
        
        self._spooled.seek(0)
        self._fully_read = True
        return self._size
    
    def read(self, size: int = -1) -> bytes:
        """Synchronous read from spooled file."""
        return self._spooled.read(size)
    
    def seek(self, pos: int):
        """Seek in spooled file."""
        self._spooled.seek(pos)
    
    def close(self):
        """Clean up resources."""
        self._spooled.close()
    
    @property
    def size(self) -> int:
        return self._size
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
```

### 3. Update `__init__` signature

```python
def __init__(
    self, 
    params: dict, 
    app: Union[Flask, FastAPI, None] = None, 
    worker_ctx=None, 
    flask_app=None  # Deprecated
):
    self._params = params
    
    # Backwards compatibility for flask_app parameter
    if flask_app is not None:
        import warnings
        warnings.warn(
            "The 'flask_app' parameter is deprecated. Use 'app' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.app = flask_app
    else:
        self.app = app
    
    self._worker_ctx = worker_ctx
    self._executor: Optional[ThreadPoolExecutor] = None
    self._executor_shutdown = False
    
    # ... rest of initialization ...
```

### 4. Executor Management with Backpressure

> ⚠️ **CRITICAL FIX:** The executor must have backpressure to prevent deadlocks when
> concurrent requests exceed executor capacity.

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Callable, TypeVar
from functools import partial

T = TypeVar("T")


class ExecutorWithBackpressure:
    """
    ThreadPoolExecutor wrapper with asyncio-based backpressure.
    
    Prevents deadlocks by limiting the number of concurrent tasks
    submitted to the executor. When the limit is reached, new requests
    wait asynchronously (non-blocking) until a slot becomes available.
    
    Key differences from raw ThreadPoolExecutor:
    - Semaphore limits concurrent submissions (not just worker threads)
    - Async waiting prevents event loop blocking
    - Timeout support for queued requests
    - Metrics for monitoring queue depth
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        max_queue_depth: int = 100,
        queue_timeout: float = 30.0,
        thread_name_prefix: str = "drum-predict-"
    ):
        """
        Initialize executor with backpressure.
        
        Args:
            max_workers: Number of threads in the pool
            max_queue_depth: Maximum number of pending tasks (backpressure limit)
            queue_timeout: Timeout for waiting in queue (seconds)
            thread_name_prefix: Prefix for thread names
        """
        self._max_workers = max_workers
        self._max_queue_depth = max_queue_depth
        self._queue_timeout = queue_timeout
        
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix
        )
        
        # Semaphore limits total concurrent operations (workers + queue)
        self._semaphore = asyncio.Semaphore(max_workers + max_queue_depth)
        
        # Track metrics
        self._pending_count = 0
        self._rejected_count = 0
        self._completed_count = 0
        self._lock = asyncio.Lock()
        
        self._shutdown = False
        
        logger.info(
            "Created ExecutorWithBackpressure: workers=%d, max_queue=%d, timeout=%.1fs",
            max_workers, max_queue_depth, queue_timeout
        )
    
    async def run(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Run a sync function in the thread pool with backpressure.
        
        Args:
            func: Synchronous function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the function
            
        Raises:
            asyncio.TimeoutError: If waiting in queue exceeds timeout
            RuntimeError: If executor is shut down
        """
        if self._shutdown:
            raise RuntimeError("Executor is shut down")
        
        # Try to acquire semaphore with timeout (backpressure)
        try:
            acquired = await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self._queue_timeout
            )
        except asyncio.TimeoutError:
            async with self._lock:
                self._rejected_count += 1
            logger.warning(
                "Request rejected due to backpressure (queue full for %.1fs). "
                "pending=%d, max=%d",
                self._queue_timeout, self._pending_count, self._max_queue_depth
            )
            raise asyncio.TimeoutError(
                f"Server overloaded. Queue full for {self._queue_timeout}s. "
                f"Try again later."
            )
        
        async with self._lock:
            self._pending_count += 1
        
        try:
            # Run in thread pool
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self._executor,
                partial(func, *args, **kwargs)
            )
            
            async with self._lock:
                self._completed_count += 1
            
            return result
        finally:
            async with self._lock:
                self._pending_count -= 1
            self._semaphore.release()
    
    async def get_metrics(self) -> dict:
        """Get executor metrics for monitoring."""
        async with self._lock:
            return {
                "executor_workers": self._max_workers,
                "executor_max_queue": self._max_queue_depth,
                "executor_pending": self._pending_count,
                "executor_queue_available": self._semaphore._value,
                "executor_completed": self._completed_count,
                "executor_rejected": self._rejected_count,
            }
    
    def shutdown(self, wait: bool = True, cancel_futures: bool = False):
        """Shutdown the executor."""
        self._shutdown = True
        logger.info("Shutting down ExecutorWithBackpressure...")
        
        import sys
        if sys.version_info >= (3, 9):
            self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)
        else:
            self._executor.shutdown(wait=wait)
        
        logger.debug("ExecutorWithBackpressure shut down successfully")


def _get_executor(self) -> ExecutorWithBackpressure:
    """Get or create thread pool executor with backpressure for sync operations."""
    if self._executor is None:
        from datarobot_drum import RuntimeParameters
        
        # Calculate optimal workers based on uvicorn workers
        uvicorn_workers = int(os.environ.get("UVICORN_WORKERS", 1))
        
        # Default: 2x uvicorn workers, minimum 4
        default_workers = max(4, uvicorn_workers * 2)
        
        if RuntimeParameters.has("DRUM_FASTAPI_EXECUTOR_WORKERS"):
            workers = int(RuntimeParameters.get("DRUM_FASTAPI_EXECUTOR_WORKERS"))
        else:
            workers = default_workers
        
        # Queue depth: allow some queuing but not unbounded
        max_queue_depth = workers * 10  # 10 pending per worker
        if RuntimeParameters.has("DRUM_FASTAPI_EXECUTOR_QUEUE_DEPTH"):
            max_queue_depth = int(RuntimeParameters.get("DRUM_FASTAPI_EXECUTOR_QUEUE_DEPTH"))
        
        # Queue timeout
        queue_timeout = 30.0
        if RuntimeParameters.has("DRUM_FASTAPI_EXECUTOR_QUEUE_TIMEOUT"):
            queue_timeout = float(RuntimeParameters.get("DRUM_FASTAPI_EXECUTOR_QUEUE_TIMEOUT"))
        
        self._executor = ExecutorWithBackpressure(
            max_workers=workers,
            max_queue_depth=max_queue_depth,
            queue_timeout=queue_timeout,
            thread_name_prefix="drum-predict-"
        )
        
        # Register cleanup with worker context
        if self._worker_ctx:
            self._worker_ctx.defer_cleanup(
                lambda: self._executor.shutdown(wait=True, cancel_futures=True),
                order=100,  # High priority - shutdown early
                desc="ExecutorWithBackpressure shutdown"
            )
        
        logger.info(
            "Created ExecutorWithBackpressure: workers=%d, queue=%d, timeout=%.1fs",
            workers, max_queue_depth, queue_timeout
        )
    
    return self._executor
```

### Backpressure Configuration

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `DRUM_FASTAPI_EXECUTOR_WORKERS` | Thread pool size | `max(4, uvicorn_workers * 2)` | CPU cores for CPU-bound, 2-4x for I/O-bound |
| `DRUM_FASTAPI_EXECUTOR_QUEUE_DEPTH` | Max pending requests | `workers * 10` | Adjust based on latency requirements |
| `DRUM_FASTAPI_EXECUTOR_QUEUE_TIMEOUT` | Queue wait timeout (sec) | `30.0` | Match client timeout |

### Why Backpressure is Critical

Without backpressure:
```
8 uvicorn workers × 100 concurrent connections = 800 tasks
4 executor threads = DEADLOCK (796 tasks waiting forever)
```

With backpressure:
```
4 executor threads + 40 queue slots = 44 max concurrent
Task #45 waits async (non-blocking) or gets 503 after timeout
```

### 5. Refactored `materialize()` Method

```python
def materialize(self):
    if isinstance(self.app, FastAPI) or self._is_fastapi_mode():
        return self._materialize_fastapi()
    else:
        return self._materialize_flask()

def _is_fastapi_mode(self) -> bool:
    from datarobot_drum import RuntimeParameters
    return (
        RuntimeParameters.has("DRUM_SERVER_TYPE") 
        and str(RuntimeParameters.get("DRUM_SERVER_TYPE")).lower() in ["fastapi", "uvicorn"]
    )
```

### 6. FastAPI Routes with Streaming Support

```python
def _materialize_fastapi(self):
    from datarobot_drum.drum.server import get_fastapi_app
    from datarobot_drum.drum.fastapi.extensions import load_fastapi_extensions
    
    router = APIRouter()
    
    @router.get("/")
    @router.get("/ping")
    @router.get("/ping/")
    async def ping():
        if hasattr(self._predictor, "liveness_probe"):
            return self._predictor.liveness_probe()
        return {"message": "OK"}
    
    @router.get("/livez")
    async def livez():
        """Kubernetes liveness probe."""
        return {"status": "alive"}
    
    @router.get("/readyz")
    async def readyz():
        """Kubernetes readiness probe."""
        if not self._worker_ctx or not self._worker_ctx.is_running:
            return JSONResponse(
                status_code=503,
                content={"status": "not ready", "reason": "worker not running"}
            )
        if hasattr(self._predictor, "readiness_probe"):
            result = self._predictor.readiness_probe()
            if result.get("status") != "ok":
                return JSONResponse(status_code=503, content=result)
        return {"status": "ready"}
    
    @router.get("/capabilities")
    @router.get("/capabilities/")
    async def capabilities():
        return self.make_capabilities()
    
    @router.get("/info")
    @router.get("/info/")
    async def info():
        model_info = self._predictor.model_info()
        model_info.update({
            ModelInfoKeys.LANGUAGE: self._run_language.value,
            ModelInfoKeys.DRUM_VERSION: drum_version,
            ModelInfoKeys.DRUM_SERVER: "fastapi",
            ModelInfoKeys.MODEL_METADATA: read_model_metadata_yaml(self._code_dir)
        })
        return model_info
    
    @router.get("/health/")
    async def health():
        if hasattr(self._predictor, "readiness_probe"):
            return self._predictor.readiness_probe()
        return {"message": "OK"}
    
    @router.get("/stats/")
    async def stats():
        ret_dict = self._resource_monitor.collect_resources_info()
        
        # Add backpressure metrics if available
        if self._worker_ctx and hasattr(self._worker_ctx, "get_backpressure_metrics"):
            ret_dict.update(self._worker_ctx.get_backpressure_metrics())
        
        # Add executor metrics (backpressure monitoring)
        if self._executor:
            executor_metrics = await self._executor.get_metrics()
            ret_dict.update(executor_metrics)
        
        return ret_dict

    @router.post("/predictions/")
    @router.post("/predict/")
    @router.post("/invocations")
    async def predict(request: Request):
        return await self._handle_predict_request(request, self.do_predict_structured)
    
    @router.post("/transform/")
    async def transform(request: Request):
        return await self._handle_predict_request(request, self.do_transform)
    
    @router.post("/predictionsUnstructured/")
    @router.post("/predictUnstructured/")
    async def predict_unstructured(request: Request):
        return await self._handle_predict_request(request, self.do_predict_unstructured)
    
    # Streaming prediction endpoint
    @router.post("/predictionsStream/")
    async def predict_stream(request: Request):
        """Streaming prediction endpoint for large responses."""
        return await self._handle_streaming_predict(request)

    # Register router
    app = get_fastapi_app(router, self.app)
    
    # Setup lifecycle
    self._setup_fastapi_lifecycle(app)
    
    return []
```

### 7. Memory-Efficient Request Handler with Backpressure

```python
async def _handle_predict_request(
    self, 
    request: Request, 
    handler_func: Callable
) -> Response:
    """
    Handle prediction request with memory-efficient file handling and backpressure.
    
    Features:
    - Memory-efficient: small payloads in memory, large ones spooled to disk
    - Backpressure: rejects requests with 503 when server is overloaded
    - Timeout: cancels requests that take too long in queue
    """
    # Update activity for stdout flusher
    if hasattr(self, '_stdout_flusher'):
        self._stdout_flusher.set_last_activity_time()
    
    # Parse request with memory-efficient handling
    request_data = await self._parse_request_efficiently(request)
    
    try:
        # OTel tracing context
        from datarobot_drum.drum.common import otel_context, extract_request_headers
        tracer = trace.get_tracer(__name__)
        
        with otel_context(tracer, "drum.invocations", request.headers) as span:
            span.set_attributes(extract_request_headers(dict(request.headers)))
            
            self._pre_predict_and_transform()
            
            try:
                # Run sync prediction with backpressure protection
                executor = self._get_executor()
                
                try:
                    response_data, status_code = await executor.run(
                        handler_func, 
                        request_data=request_data
                    )
                except asyncio.TimeoutError as e:
                    # Server overloaded - return 503 Service Unavailable
                    logger.warning(
                        "Request rejected due to backpressure: %s",
                        str(e)
                    )
                    return JSONResponse(
                        status_code=503,
                        content={
                            "message": "Server overloaded. Please retry later.",
                            "error": "SERVICE_UNAVAILABLE",
                            "retry_after": 5
                        },
                        headers={"Retry-After": "5"}
                    )
                
                return self._create_response(response_data, status_code)
            finally:
                self._post_predict_and_transform()
    
    finally:
        # Cleanup spooled files
        self._cleanup_request_data(request_data)


async def _parse_request_efficiently(self, request: Request) -> Dict[str, Any]:
    """
    Parse request with memory-efficient handling.
    
    Strategy:
    - Small bodies (< 10MB): keep in memory
    - Large bodies: spool to disk
    - Multipart files: use SpooledUploadFile
    """
    content_type = request.headers.get("content-type", "")
    content_length = int(request.headers.get("content-length", 0))
    
    request_data = {
        "headers": dict(request.headers),
        "content_type": content_type,
        "body": None,
        "files": {},
        "_spooled_files": [],  # Track for cleanup
    }
    
    if "multipart/form-data" in content_type:
        form = await request.form()
        
        for key, value in form.items():
            if isinstance(value, UploadFile):
                # Use spooled file for memory efficiency
                spooled = SpooledUploadFile(value)
                await spooled.read_to_spool()
                
                request_data["files"][key] = {
                    "content": spooled,  # Pass spooled file, not bytes
                    "filename": value.filename,
                    "content_type": value.content_type,
                    "size": spooled.size,
                }
                request_data["_spooled_files"].append(spooled)
                
                logger.debug(
                    "Processed file %s: %d bytes (spooled: %s)",
                    key, spooled.size, spooled.size > SPOOL_MAX_SIZE
                )
            else:
                request_data.setdefault("form", {})[key] = value
    
    elif content_length > SPOOL_MAX_SIZE:
        # Large non-multipart body - spool to disk
        spooled = SpooledTemporaryFile(max_size=SPOOL_MAX_SIZE, mode='w+b')
        
        async for chunk in request.stream():
            spooled.write(chunk)
        
        spooled.seek(0)
        request_data["body"] = spooled
        request_data["_spooled_files"].append(spooled)
        
        logger.debug("Spooled large request body: %d bytes", content_length)
    
    else:
        # Small body - keep in memory
        request_data["body"] = await request.body()
    
    return request_data


def _cleanup_request_data(self, request_data: Dict[str, Any]):
    """Clean up spooled files after request processing."""
    for spooled in request_data.get("_spooled_files", []):
        try:
            if hasattr(spooled, "close"):
                spooled.close()
        except Exception as e:
            logger.warning("Error closing spooled file: %s", e)
```

### 8. Streaming Response Support

```python
async def _handle_streaming_predict(self, request: Request) -> StreamingResponse:
    """
    Handle streaming prediction for large responses.
    
    Useful for:
    - Large batch predictions
    - Generative model outputs
    - Real-time streaming
    """
    request_data = await self._parse_request_efficiently(request)
    
    async def generate() -> AsyncIterator[bytes]:
        try:
            loop = asyncio.get_running_loop()
            
            # Get prediction iterator from predictor
            if hasattr(self._predictor, "predict_stream"):
                async for chunk in self._predictor.predict_stream(request_data):
                    yield chunk
            else:
                # Fallback: run regular prediction and yield all at once
                response_data, _ = await loop.run_in_executor(
                    self._get_executor(),
                    partial(self.do_predict_structured, request_data=request_data)
                )
                yield response_data
        finally:
            self._cleanup_request_data(request_data)
    
    return StreamingResponse(
        generate(),
        media_type="application/octet-stream",
        headers={"X-Streaming": "true"}
    )


def _create_response(self, data: Any, status_code: int) -> Response:
    """Create appropriate response based on data type."""
    if isinstance(data, bytes):
        return Response(
            content=data,
            status_code=status_code,
            media_type="application/octet-stream"
        )
    elif isinstance(data, str):
        return Response(
            content=data,
            status_code=status_code,
            media_type="text/plain"
        )
    else:
        return JSONResponse(content=data, status_code=status_code)
```

### 9. Lifecycle Setup

```python
def _setup_fastapi_lifecycle(self, app: FastAPI):
    """Setup middlewares, extensions, and cleanup hooks."""
    from datarobot_drum.drum.fastapi.extensions import load_fastapi_extensions
    from datarobot_drum.drum.fastapi.middleware import StdoutFlusherMiddleware
    
    # Store config reference
    app.state.prediction_server = self
    
    # Setup stdout flusher if available
    if hasattr(self, '_stdout_flusher') and self._stdout_flusher:
        app.add_middleware(StdoutFlusherMiddleware, flusher=self._stdout_flusher)
        
        if self._worker_ctx:
            self._worker_ctx.add_thread(
                self._stdout_flusher._flusher_thread, 
                name="StdoutFlusher"
            )
            self._stdout_flusher.start()
    
    # Load user extensions
    load_fastapi_extensions(app, self._code_dir)
    
    # Start watchdog if enabled
    self._start_watchdog_if_enabled()
    
    # Ensure executor is created
    self._get_executor()
    
    logger.info("FastAPI lifecycle setup complete")
```

## Runtime Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `DRUM_SERVER_TYPE` | Server type: "flask", "gunicorn", or "fastapi" | "flask" |
| `DRUM_FASTAPI_EXECUTOR_WORKERS` | Thread pool size for sync operations | 4 |
| `DRUM_FASTAPI_MAX_UPLOAD_SIZE` | Max request body size | 100MB |

## Memory Management Summary

| Payload Size | Handling | Location |
|--------------|----------|----------|
| < 10MB | In-memory bytes | RAM |
| >= 10MB | SpooledTemporaryFile | Disk (auto) |
| Streaming | AsyncIterator | Chunked |

## Notes

- **No Logic Duplication**: Routes call shared handler methods
- **Resource Safety**: Executor and spooled files cleaned up via worker_ctx
- **Memory Efficiency**: Large files automatically spooled to disk
- **Streaming Support**: New `/predictionsStream/` endpoint for large responses
- **K8s Ready**: `/livez` and `/readyz` endpoints added
