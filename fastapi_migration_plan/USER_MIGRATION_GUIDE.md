# User Migration Guide: From Flask to FastAPI

This guide helps you migrate your custom DRUM extensions from Flask (`custom_flask.py`) to FastAPI (`custom_fastapi.py`).

## Overview

DRUM is transitioning from a Flask-based production server to a modern FastAPI-based server. While we provide a dual-support period, we recommend migrating your custom extensions to take advantage of better performance and async support.

## Key Differences

| Feature | Flask (`custom_flask.py`) | FastAPI (`custom_fastapi.py`) |
|---------|-------------------------|---------------------------|
| **Hook File** | `custom_flask.py` | `custom_fastapi.py` |
| **Registration** | `init_app(app: Flask)` | `init_app(app: FastAPI)` |
| **Middleware** | `@app.before_request` | `BaseHTTPMiddleware` |
| **Routing** | `Blueprint` | `APIRouter` |
| **Response** | `flask.jsonify()` | `fastapi.responses.JSONResponse` |

## Migration Examples

### 1. Authentication Middleware

#### Flask Version (`custom_flask.py`)
```python
from flask import request, jsonify

def init_app(app):
    @app.before_request
    def check_auth():
        if request.endpoint == 'model_api.ping':
            return
        
        token = request.headers.get("X-Auth-Token")
        if token != "secret":
            return jsonify({"message": "Unauthorized"}), 401
```

#### FastAPI Version (`custom_fastapi.py`)
```python
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path.rstrip("/") in ["/ping", "/"]:
            return await call_next(request)
        
        token = request.headers.get("X-Auth-Token")
        if token != "secret":
            return JSONResponse(
                content={"message": "Unauthorized"},
                status_code=401
            )
        return await call_next(request)

def init_app(app):
    app.add_middleware(AuthMiddleware)
```

### 2. Adding Custom Routes

#### Flask Version (`custom_flask.py`)
```python
from flask import Blueprint, jsonify

custom_bp = Blueprint("custom", __name__)

@custom_bp.route("/custom/status")
def status():
    return jsonify({"status": "ok"})

def init_app(app):
    app.register_blueprint(custom_bp)
```

#### FastAPI Version (`custom_fastapi.py`)
```python
from fastapi import APIRouter

router = APIRouter()

@router.get("/custom/status")
async def status():
    return {"status": "ok"}

def init_app(app):
    app.include_router(router)
```

## Testing Your Migration

You can test your migrated extension by setting the `DRUM_SERVER_TYPE` environment variable:

```bash
export MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE=fastapi
drum server --code-dir your_model_dir --address localhost:8080 --target-type regression
```

Check the logs for:
- `Detected custom_fastapi.py .. trying to load FastAPI extensions`
- `Successfully loaded FastAPI extensions`

## New Runtime Parameters for FastAPI

When using `DRUM_SERVER_TYPE=fastapi`, several new parameters are available for fine-tuning the server:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `DRUM_FASTAPI_EXECUTOR_WORKERS` | Number of threads in the pool for sync prediction tasks. | 4 |
| `DRUM_FASTAPI_MAX_UPLOAD_SIZE` | Maximum allowed size for request body in bytes. | 104857600 (100MB) |
| `DRUM_FASTAPI_ENABLE_DOCS` | Enable FastAPI automatic documentation (Swagger UI at `/docs`). | false |
| `DRUM_UVICORN_LOOP` | Event loop implementation to use: `auto`, `asyncio`, or `uvloop`. | `auto` |
| `DRUM_CORS_ENABLED` | Enable Cross-Origin Resource Sharing (CORS) middleware. | false |
| `DRUM_CORS_ORIGINS` | Comma-separated list of allowed origins (e.g., `http://localhost:3000,https://app.datarobot.com`). | `*` |

## Frequently Asked Questions

### Can I keep using `custom_flask.py`?
Yes, for now. DRUM will continue to support Flask-based extensions during the transition period. However, we recommend migrating to FastAPI for better performance.

### What happens if I have both files?
If both `custom_flask.py` and `custom_fastapi.py` are present, DRUM will load the one corresponding to the selected `DRUM_SERVER_TYPE`.

### Do I need to use `async/await`?
In FastAPI, you can use either sync or async functions for your routes. For middleware `dispatch` method, it must be `async`.

## Troubleshooting & Common Pitfalls

### 1. Blocking the Event Loop
If you use synchronous blocking calls (like `requests.get()`, `time.sleep()`, or heavy CPU computation) inside an `async def` route or middleware, you will block the entire server from handling other requests.

**Solution**: 
- Use `httpx.AsyncClient` instead of `requests`.
- Use `await asyncio.sleep()` instead of `time.sleep()`.
- If you must use a sync library, define your route as `def` (without `async`)—FastAPI will automatically run it in a thread pool.
- **Heavy CPU tasks**: For heavy computation in `custom_fastapi.py`, prefer offloading to `anyio.to_thread.run_sync` or just use a sync `def`.

### 2. Accessing Request Body Multiple Times
In FastAPI, the request body is a stream. Once consumed, it cannot be read again easily unless cached.

**Solution**:
DRUM's internal middleware pre-fetches the body into `request.state.body`. Use that instead of `await request.body()` if you need to access it in multiple places. Similarly, files are available in `request.state.files`.

### 3. Middleware Order
FastAPI executes middleware in reverse order of addition.

**Solution**:
If you need your middleware to run *before* DRUM's internal logic, add it *after* DRUM has initialized the app (which is what `init_app` does).
