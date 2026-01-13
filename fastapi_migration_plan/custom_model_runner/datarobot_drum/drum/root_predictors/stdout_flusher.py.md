# Plan: custom_model_runner/datarobot_drum/drum/root_predictors/stdout_flusher.py

Integrate `StdoutFlusher` with FastAPI using middleware.

## Overview

`StdoutFlusher` is used to ensure that logs are flushed to stdout even if the server is not under heavy load. In the Flask implementation, this is often handled in an `after_request` hook. For FastAPI, we will use middleware to update the last activity time.

## Proposed Implementation

### Middleware in `server.py`

```python
class StdoutFlusherMiddleware(BaseHTTPMiddleware):
    """
    Middleware to update last activity time for StdoutFlusher.
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
```

### Integration in `PredictionServer` (`prediction_server.py`)

```python
def _materialize_fastapi(self):
    # ... existing logic to create app and router ...

    # 1. Create StdoutFlusher if needed
    if self._params.get("with_stdout_flushing"):
        self._stdout_flusher = StdoutFlusher()
        
        # 2. Add middleware to FastAPI app
        app.add_middleware(StdoutFlusherMiddleware, flusher=self._stdout_flusher)
        
        # 3. Register for cleanup in WorkerCtx
        if self._worker_ctx:
            self._worker_ctx.add_thread(
                self._stdout_flusher._flusher_thread,
                name="StdoutFlusher"
            )
            # Ensure it stops before other resources
            self._worker_ctx.defer_cleanup(
                self._stdout_flusher.stop,
                order=50,
                desc="StdoutFlusher.stop()"
            )
            
        # 4. Start the flusher
        self._stdout_flusher.start()
```

## Key Changes

1.  **Middleware**: `StdoutFlusherMiddleware` replaces the `after_request` hook used in Flask.
2.  **Worker Context Integration**: The flusher thread is now registered with `FastAPIWorkerCtx` for graceful shutdown, mirroring the Gunicorn implementation but using the new async-compatible context.
3.  **Explicit Stop**: Added `defer_cleanup` with priority `50` to ensure logs are flushed before the process exits.
4.  **Running Flag Synchronization**: The `StdoutFlusher`'s internal `_running` flag must be synchronized with the `FastAPIWorkerCtx.running()` state. When `stop()` is called by the context, it sets `_running = False`, causing the flusher thread to exit its loop after the final flush.

## Notes

- The flusher thread is a daemon thread by default (`setDaemon(True)` in `__init__`), but explicit joining in `stop()` is preferred for reliable log flushing.
- If multiple workers are used, each worker will have its own `StdoutFlusher` instance.
