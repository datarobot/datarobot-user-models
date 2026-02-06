# Removal Plan: custom_model_runner/datarobot_drum/drum/root_predictors/stdout_flusher.py

Integrate `StdoutFlusher` with FastAPI using middleware.

## Current State

While the file doesn't have direct Flask imports, it's designed to work with Flask's `after_request` hook integration in `prediction_server.py`.

## Actions

### Phase 1: Add Middleware Support

1. **Create middleware wrapper** (in `fastapi/server.py`):
```python
class StdoutFlusherMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, flusher=None):
        super().__init__(app)
        self.flusher = flusher

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if self.flusher:
            self.flusher.set_last_activity_time()
        return response
```

2. **Integration in `prediction_server.py`**:
```python
def _materialize_fastapi(self):
    if self._params.get("with_stdout_flushing"):
        self._stdout_flusher = StdoutFlusher()
        app.add_middleware(StdoutFlusherMiddleware, flusher=self._stdout_flusher)
        self._stdout_flusher.start()
```

### Phase 2: Worker Context Integration

Register flusher with `FastAPIWorkerCtx` for graceful shutdown:
```python
if self._worker_ctx:
    self._worker_ctx.defer_cleanup(
        self._stdout_flusher.stop,
        order=50,
        desc="StdoutFlusher.stop()"
    )
```

## Notes

- Middleware replaces `after_request` hook
- Flusher thread lifecycle tied to worker context
- Each Uvicorn worker has its own StdoutFlusher instance
