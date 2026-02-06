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

---

## Pydantic v2 Migration

DRUM's FastAPI server uses **Pydantic v2** (>=2.5.0). If your custom model code uses Pydantic v1 patterns, you'll need to update them.

### Why Pydantic v2?

- **Performance:** 5-50x faster validation
- **Better typing:** Improved type hints and IDE support
- **Future-proof:** FastAPI 1.0 will require Pydantic v2

### Python Version Requirements

| Python Version | Support |
|----------------|---------|
| 3.7, 3.8 | Use `DRUM_SERVER_TYPE=flask` |
| 3.9+ | FastAPI supported |
| 3.11, 3.12 | Recommended |

### Migration Patterns

| v1 Pattern | v2 Pattern |
|------------|------------|
| `class Config:` | `model_config = ConfigDict(...)` |
| `.dict()` | `.model_dump()` |
| `.json()` | `.model_dump_json()` |
| `@validator` | `@field_validator` |
| `@root_validator` | `@model_validator` |
| `constr(regex=...)` | `Annotated[str, Field(pattern=...)]` |
| `constr(min_length=1)` | `Annotated[str, Field(min_length=1)]` |
| `conlist(Item, min_items=1)` | `Annotated[List[Item], Field(min_length=1)]` |
| `.parse_raw()` | `.model_validate_json()` |
| `.parse_obj()` | `.model_validate()` |
| `__fields__` | `model_fields` |
| `.schema()` | `.model_json_schema()` |

### Example: Pydantic Model Migration

**Before (Pydantic v1):**
```python
from pydantic import BaseModel, validator, constr

class PredictionInput(BaseModel):
    feature_name: constr(min_length=1)
    value: float
    
    class Config:
        extra = "ignore"
    
    @validator("value")
    def value_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("value must be positive")
        return v
```

**After (Pydantic v2):**
```python
from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field, field_validator

class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    feature_name: Annotated[str, Field(min_length=1)]
    value: float
    
    @field_validator("value")
    @classmethod
    def value_must_be_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("value must be positive")
        return v
```

### Required Library Versions

If using these libraries with Pydantic v2:

| Library | Minimum Version | Notes |
|---------|-----------------|-------|
| `langchain` | >= 0.1.0 | Pydantic v2 support |
| `mlflow` | >= 2.9.0 | Pydantic v2 support |
| `ray[serve]` | >= 2.8.0 | Pydantic v2 support |
| `datarobot` SDK | Any | Compatible |
| `transformers` | Any | No conflict |

### Checking Compatibility

Run this script to check your environment:

```python
from importlib.metadata import version, PackageNotFoundError

def check_pydantic_v2():
    try:
        pydantic_ver = version("pydantic")
        major = int(pydantic_ver.split(".")[0])
        if major < 2:
            print(f"⚠️ Pydantic {pydantic_ver} is v1. Upgrade: pip install 'pydantic>=2.5.0'")
            return False
        print(f"✅ Pydantic {pydantic_ver} (v2)")
        return True
    except PackageNotFoundError:
        print("❌ Pydantic not installed")
        return False

check_pydantic_v2()
```

---

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

### 3. Stateful Middleware (Rate Limiter Example)

This example shows how to migrate a stateful middleware that maintains counters across requests.

#### Flask Version (`custom_flask.py`)
```python
import time
from flask import request, jsonify, g
from collections import defaultdict
import threading

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.request_counts = defaultdict(list)
        self._lock = threading.Lock()
    
    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        minute_ago = now - 60
        
        with self._lock:
            # Clean old entries
            self.request_counts[client_ip] = [
                t for t in self.request_counts[client_ip] if t > minute_ago
            ]
            
            if len(self.request_counts[client_ip]) >= self.requests_per_minute:
                return False
            
            self.request_counts[client_ip].append(now)
            return True
    
    def get_stats(self) -> dict:
        with self._lock:
            return {
                "active_clients": len(self.request_counts),
                "total_tracked_requests": sum(len(v) for v in self.request_counts.values())
            }

# Global instance
rate_limiter = RateLimiter(requests_per_minute=100)

def init_app(app):
    @app.before_request
    def check_rate_limit():
        if request.endpoint in ['model_api.ping', 'model_api.health']:
            return
        
        client_ip = request.remote_addr
        if not rate_limiter.is_allowed(client_ip):
            return jsonify({"message": "Rate limit exceeded"}), 429
    
    # Add stats endpoint
    @app.route("/custom/rate-limit-stats")
    def rate_limit_stats():
        return jsonify(rate_limiter.get_stats())
```

#### FastAPI Version (`custom_fastapi.py`)
```python
import time
import asyncio
from collections import defaultdict
from fastapi import Request, APIRouter
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

class RateLimiter:
    """Async-safe rate limiter with state."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.request_counts: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        minute_ago = now - 60
        
        async with self._lock:
            # Clean old entries
            self.request_counts[client_ip] = [
                t for t in self.request_counts[client_ip] if t > minute_ago
            ]
            
            if len(self.request_counts[client_ip]) >= self.requests_per_minute:
                return False
            
            self.request_counts[client_ip].append(now)
            return True
    
    async def get_stats(self) -> dict:
        async with self._lock:
            return {
                "active_clients": len(self.request_counts),
                "total_tracked_requests": sum(len(v) for v in self.request_counts.values())
            }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware that uses the rate limiter."""
    
    # Store rate limiter as class attribute so it persists across requests
    rate_limiter: RateLimiter = None
    
    def __init__(self, app, rate_limiter: RateLimiter):
        super().__init__(app)
        RateLimitMiddleware.rate_limiter = rate_limiter
    
    async def dispatch(self, request: Request, call_next):
        # Skip health endpoints
        if request.url.path.rstrip("/") in ["/ping", "/health", "/livez", "/readyz"]:
            return await call_next(request)
        
        client_ip = request.client.host if request.client else "unknown"
        
        if not await self.rate_limiter.is_allowed(client_ip):
            return JSONResponse(
                content={"message": "Rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": "60"}
            )
        
        return await call_next(request)


# Router for stats endpoint
router = APIRouter()

@router.get("/custom/rate-limit-stats")
async def rate_limit_stats():
    """Expose rate limiter statistics."""
    if RateLimitMiddleware.rate_limiter:
        stats = await RateLimitMiddleware.rate_limiter.get_stats()
        return stats
    return {"message": "Rate limiter not initialized"}


def init_app(app):
    # Create rate limiter instance
    rate_limiter = RateLimiter(requests_per_minute=100)
    
    # Add middleware with state
    app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)
    
    # Add stats router
    app.include_router(router)
```

**Key differences for stateful middleware:**

| Aspect | Flask | FastAPI |
|--------|-------|---------|
| Lock type | `threading.Lock` | `asyncio.Lock` |
| State storage | Global variable | Class attribute or `app.state` |
| Async safety | Thread-safe | Async-safe |
| Access in routes | Global import | Via middleware class attribute or `request.app.state` |

### 4. Using `app.state` for Shared State

Another pattern for sharing state between middleware and routes:

```python
# custom_fastapi.py
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Access shared state via app.state
        request.app.state.request_count += 1
        return await call_next(request)

def init_app(app):
    # Initialize shared state
    app.state.request_count = 0
    app.state.custom_config = {"feature_flag": True}
    
    # Add middleware
    app.add_middleware(MetricsMiddleware)
    
    # Routes can access state
    @app.get("/custom/metrics")
    async def metrics(request: Request):
        return {
            "request_count": request.app.state.request_count,
            "config": request.app.state.custom_config
        }
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

### 4. Threading vs Asyncio Locks
If you're migrating stateful middleware, you must use `asyncio.Lock()` instead of `threading.Lock()`.

**Problem:**
```python
# WRONG - will block the event loop!
import threading
lock = threading.Lock()

class BadMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        with lock:  # This blocks!
            # ...
```

**Solution:**
```python
# CORRECT - async-safe
import asyncio
lock = asyncio.Lock()

class GoodMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        async with lock:  # Non-blocking
            # ...
```

### 5. Windows Compatibility
On Windows, `uvloop` is not available. The server will automatically fall back to `asyncio`.

**Symptom:** Warning in logs: "uvloop is not available on Windows"

**Solution:** This is expected behavior. No action needed. Performance on Windows may be slightly lower than Linux/macOS.

### 6. Pydantic v2 Migration
DRUM's FastAPI server requires Pydantic v2 (>=2.5.0). If your custom code uses Pydantic v1 patterns, you must update them.

**Problem:** `ImportError`, `ValidationError`, or deprecated warnings related to Pydantic

**Solution:**
- Check your Pydantic version: `pip show pydantic`
- Update v1 patterns to v2 (see "Pydantic v2 Migration" section above)
- Ensure ML libraries are updated (langchain >= 0.1.0, mlflow >= 2.9.0, ray >= 2.8.0)
- See [COMPATIBILITY_MATRIX.md](COMPATIBILITY_MATRIX.md) for detailed compatibility info

**Note:** The `from pydantic.v1 import BaseModel` shim is available but not recommended for new code. Migrate fully to v2 patterns for best performance.

---

## Complete Migration Checklist

Use this checklist when migrating your `custom_flask.py` to `custom_fastapi.py`:

### Pre-Migration

- [ ] Verify Python version >= 3.9 (3.11 or 3.12 recommended)
- [ ] Identify all Flask-specific imports in your code
- [ ] List all `@app.before_request` and `@app.after_request` hooks
- [ ] Document all `Blueprint` routes and their purposes
- [ ] Check for any global state (counters, caches, etc.)
- [ ] Review threading locks (need to convert to asyncio.Lock)
- [ ] Identify Pydantic v1 patterns that need migration

### Pydantic v2 Migration Steps

- [ ] Update Pydantic import: `from pydantic import BaseModel, ConfigDict, Field`
- [ ] Replace `class Config:` with `model_config = ConfigDict(...)`
- [ ] Replace `constr(min_length=1)` with `Annotated[str, Field(min_length=1)]`
- [ ] Replace `conlist(Item)` with `Annotated[List[Item], Field(...)]`
- [ ] Replace `.dict()` with `.model_dump()`
- [ ] Replace `.json()` with `.model_dump_json()`
- [ ] Replace `.parse_raw()` with `.model_validate_json()`
- [ ] Replace `@validator` with `@field_validator` (add `@classmethod`)
- [ ] Replace `@root_validator` with `@model_validator`
- [ ] Update ML library versions if needed (langchain >= 0.1.0, mlflow >= 2.9.0)

### Flask to FastAPI Migration Steps

- [ ] Create new `custom_fastapi.py` file
- [ ] Convert imports:
  - [ ] `flask.request` → `fastapi.Request`
  - [ ] `flask.jsonify` → `fastapi.responses.JSONResponse`
  - [ ] `flask.Blueprint` → `fastapi.APIRouter`
  - [ ] `threading.Lock` → `asyncio.Lock`
- [ ] Convert `@app.before_request` to `BaseHTTPMiddleware`
- [ ] Convert `Blueprint` routes to `APIRouter`
- [ ] Add `async` keyword to middleware `dispatch` method
- [ ] Update request/response handling for async
- [ ] Test with `DRUM_SERVER_TYPE=fastapi`

### Post-Migration Validation

- [ ] All endpoints return same responses as Flask
- [ ] Authentication/authorization works correctly
- [ ] Custom routes are accessible
- [ ] No blocking calls in async code
- [ ] Memory usage is stable under load
- [ ] Pydantic models validate correctly

---

## API Reference: Flask → FastAPI

### Request Object

| Flask | FastAPI | Notes |
|-------|---------|-------|
| `request.method` | `request.method` | Same |
| `request.path` | `request.url.path` | Different attribute |
| `request.headers` | `request.headers` | Same (immutable) |
| `request.headers.get("X-Key")` | `request.headers.get("X-Key")` | Same |
| `request.args` | `request.query_params` | Different name |
| `request.args.get("key")` | `request.query_params.get("key")` | Same pattern |
| `request.form` | `await request.form()` | Async in FastAPI! |
| `request.json` | `await request.json()` | Async in FastAPI! |
| `request.data` | `await request.body()` | Async in FastAPI! |
| `request.files` | `await request.form()` then filter `UploadFile` | Different pattern |
| `request.remote_addr` | `request.client.host` | Different path |
| `request.endpoint` | `request.url.path` | No direct equivalent |

### Response Objects

| Flask | FastAPI | Notes |
|-------|---------|-------|
| `jsonify({"key": "value"})` | `JSONResponse(content={"key": "value"})` | Or just return dict |
| `return {"key": "value"}, 200` | `return {"key": "value"}` | Status 200 is default |
| `return {"error": "msg"}, 400` | `return JSONResponse(content={"error": "msg"}, status_code=400)` | Explicit status |
| `Response(data, mimetype="text/plain")` | `Response(content=data, media_type="text/plain")` | Different params |
| `make_response(...)` | `Response(...)` | Direct construction |
| `abort(404)` | `raise HTTPException(status_code=404)` | Exception-based |

### Middleware Hooks

| Flask | FastAPI | Notes |
|-------|---------|-------|
| `@app.before_request` | `BaseHTTPMiddleware.dispatch()` (before `call_next`) | Different pattern |
| `@app.after_request` | `BaseHTTPMiddleware.dispatch()` (after `call_next`) | Different pattern |
| `@app.teardown_request` | Context manager or `try/finally` in middleware | Manual cleanup |
| `g.user = ...` | `request.state.user = ...` | Different storage |

### Example: Full Migration

**Flask (`custom_flask.py`):**
```python
from flask import Blueprint, request, jsonify, g
import time

bp = Blueprint("custom", __name__)

# Global state
request_count = 0

def init_app(app):
    @app.before_request
    def before():
        global request_count
        request_count += 1
        g.start_time = time.time()
        
        # Auth check
        if request.endpoint not in ["model_api.ping", "custom.health"]:
            token = request.headers.get("Authorization")
            if not token or not token.startswith("Bearer "):
                return jsonify({"error": "Unauthorized"}), 401
            g.user = validate_token(token[7:])
    
    @app.after_request
    def after(response):
        duration = time.time() - g.start_time
        response.headers["X-Request-Duration"] = str(duration)
        return response
    
    app.register_blueprint(bp)

@bp.route("/custom/health")
def health():
    return jsonify({"status": "ok", "requests": request_count})

@bp.route("/custom/user")
def user():
    return jsonify({"user": g.user})
```

**FastAPI (`custom_fastapi.py`):**
```python
import time
import asyncio
from fastapi import Request, APIRouter
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

router = APIRouter()

# Global state with async-safe lock
request_count = 0
_lock = asyncio.Lock()


class CustomMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        global request_count
        
        # Before request
        async with _lock:
            request_count += 1
        
        request.state.start_time = time.time()
        
        # Auth check (skip health endpoints)
        skip_auth = ["/ping", "/health", "/livez", "/readyz", "/custom/health"]
        if request.url.path.rstrip("/") not in skip_auth:
            token = request.headers.get("Authorization", "")
            if not token.startswith("Bearer "):
                return JSONResponse(
                    content={"error": "Unauthorized"},
                    status_code=401
                )
            request.state.user = validate_token(token[7:])
        
        # Call next (the actual endpoint)
        response = await call_next(request)
        
        # After request
        duration = time.time() - request.state.start_time
        response.headers["X-Request-Duration"] = str(duration)
        
        return response


@router.get("/custom/health")
async def health():
    async with _lock:
        count = request_count
    return {"status": "ok", "requests": count}


@router.get("/custom/user")
async def user(request: Request):
    return {"user": request.state.user}


def init_app(app):
    app.add_middleware(CustomMiddleware)
    app.include_router(router)
```

---

## Parity Testing

After migration, verify your FastAPI implementation produces identical results to Flask.

### Quick Parity Test Script

```python
#!/usr/bin/env python3
"""
Quick parity test for custom_flask.py → custom_fastapi.py migration.

Usage:
    python test_parity.py --flask-url http://localhost:8080 --fastapi-url http://localhost:8081
"""
import argparse
import json
import sys
from typing import Dict, Any, List, Tuple

import httpx
from deepdiff import DeepDiff


def compare_responses(
    flask_resp: httpx.Response,
    fastapi_resp: httpx.Response,
    endpoint: str
) -> Tuple[bool, str]:
    """Compare two responses and return (is_equal, diff_description)."""
    
    # Compare status codes
    if flask_resp.status_code != fastapi_resp.status_code:
        return False, f"Status mismatch: Flask={flask_resp.status_code}, FastAPI={fastapi_resp.status_code}"
    
    # Compare headers (ignore server-specific ones)
    ignore_headers = {"server", "date", "x-request-duration"}
    flask_headers = {k.lower(): v for k, v in flask_resp.headers.items() if k.lower() not in ignore_headers}
    fastapi_headers = {k.lower(): v for k, v in fastapi_resp.headers.items() if k.lower() not in ignore_headers}
    
    header_diff = DeepDiff(flask_headers, fastapi_headers, ignore_order=True)
    if header_diff:
        return False, f"Header diff: {header_diff}"
    
    # Compare body
    try:
        flask_json = flask_resp.json()
        fastapi_json = fastapi_resp.json()
        body_diff = DeepDiff(flask_json, fastapi_json, ignore_order=True, significant_digits=5)
        if body_diff:
            return False, f"Body diff: {body_diff}"
    except json.JSONDecodeError:
        if flask_resp.content != fastapi_resp.content:
            return False, f"Binary content mismatch: {len(flask_resp.content)} vs {len(fastapi_resp.content)} bytes"
    
    return True, "OK"


def run_parity_tests(flask_url: str, fastapi_url: str) -> List[Dict[str, Any]]:
    """Run parity tests on common endpoints."""
    
    test_cases = [
        # Health endpoints
        ("GET", "/ping", None, {}),
        ("GET", "/health/", None, {}),
        ("GET", "/info", None, {}),
        
        # Custom endpoints (add your own)
        ("GET", "/custom/health", None, {}),
        
        # Prediction (example - customize for your model)
        # ("POST", "/predict/", {"feature1": 1.0, "feature2": 2.0}, {"Content-Type": "application/json"}),
    ]
    
    results = []
    
    with httpx.Client(timeout=30.0) as client:
        for method, path, body, headers in test_cases:
            try:
                # Flask request
                flask_resp = client.request(
                    method, f"{flask_url}{path}",
                    json=body if body else None,
                    headers=headers
                )
                
                # FastAPI request
                fastapi_resp = client.request(
                    method, f"{fastapi_url}{path}",
                    json=body if body else None,
                    headers=headers
                )
                
                is_equal, diff = compare_responses(flask_resp, fastapi_resp, path)
                
                results.append({
                    "endpoint": f"{method} {path}",
                    "passed": is_equal,
                    "diff": diff if not is_equal else None,
                    "flask_status": flask_resp.status_code,
                    "fastapi_status": fastapi_resp.status_code,
                })
                
            except Exception as e:
                results.append({
                    "endpoint": f"{method} {path}",
                    "passed": False,
                    "diff": str(e),
                    "flask_status": None,
                    "fastapi_status": None,
                })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Parity test for Flask → FastAPI migration")
    parser.add_argument("--flask-url", required=True, help="Flask server URL")
    parser.add_argument("--fastapi-url", required=True, help="FastAPI server URL")
    args = parser.parse_args()
    
    print(f"Running parity tests...")
    print(f"  Flask:   {args.flask_url}")
    print(f"  FastAPI: {args.fastapi_url}")
    print()
    
    results = run_parity_tests(args.flask_url, args.fastapi_url)
    
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    
    for r in results:
        status = "✅" if r["passed"] else "❌"
        print(f"{status} {r['endpoint']}")
        if not r["passed"]:
            print(f"   {r['diff']}")
    
    print()
    print(f"Results: {passed}/{total} passed")
    
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
```

### Running the Parity Test

```bash
# Start Flask server
DRUM_SERVER_TYPE=flask drum server --code-dir ./model --address 0.0.0.0:8080 &

# Start FastAPI server  
DRUM_SERVER_TYPE=fastapi drum server --code-dir ./model --address 0.0.0.0:8081 &

# Run parity test
python test_parity.py --flask-url http://localhost:8080 --fastapi-url http://localhost:8081
```
