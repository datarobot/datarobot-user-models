# Plan: tests/fixtures/custom_fastapi_demo_auth.py

A test fixture to verify FastAPI extension mechanism. Mirrors `custom_flask_demo_auth.py`.

## Overview

This fixture demonstrates how to add custom middleware to the FastAPI application for authentication purposes. It is used by `test_drum_server_fastapi.py` to verify that custom FastAPI extensions are loaded correctly.

## Proposed Implementation:

```python
"""
Copyright 2024 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Custom authentication middleware.
    Checks for X-Auth header on all endpoints except /ping/.
    """
    
    async def dispatch(self, request: Request, call_next):
        # Allow ping route without auth (so test setup doesn't fail)
        if request.url.path.rstrip("/") in ["/ping", "/"]:
            return await call_next(request)
        
        # Check for X-Auth header
        auth_token = request.headers.get("X-Auth")
        
        if auth_token is None:
            return JSONResponse(
                content={"message": "Missing X-Auth header"},
                status_code=401
            )
        
        if auth_token != "t0k3n":
            return JSONResponse(
                content={"message": "Auth token is invalid"},
                status_code=401
            )
        
        return await call_next(request)


def init_app(app: FastAPI):
    """
    Initialize the FastAPI app with custom middleware.
    
    This function is called by DRUM's load_fastapi_extensions() when
    a custom_fastapi.py file is detected in the model directory.
    
    Args:
        app: The FastAPI application instance.
    """
    app.add_middleware(AuthMiddleware)
```

## Alternative Implementation Using Dependencies

For more granular control, you can also use FastAPI's dependency injection:

```python
from fastapi import FastAPI, Request, Depends, HTTPException


async def verify_auth(request: Request):
    """Dependency to verify authentication."""
    # Allow ping route without auth
    if request.url.path.rstrip("/") in ["/ping", "/"]:
        return
    
    auth_token = request.headers.get("X-Auth")
    
    if auth_token is None:
        raise HTTPException(status_code=401, detail="Missing X-Auth header")
    
    if auth_token != "t0k3n":
        raise HTTPException(status_code=401, detail="Auth token is invalid")


def init_app(app: FastAPI):
    """
    Add auth dependency to all routes.
    """
    # Add dependency to all routes
    app.dependency_overrides[verify_auth] = verify_auth
    
    # Or add as a global dependency (affects all routes)
    # This requires modifying how routes are registered
```

## Usage in Tests

The fixture file should be copied to the model directory as `custom_fastapi.py`:

```python
@pytest.fixture(scope="class")
def custom_fastapi_script(self):
    return (Path(TESTS_FIXTURES_PATH) / "custom_fastapi_demo_auth.py", "custom_fastapi.py")

@pytest.fixture(scope="class")
def custom_model_dir(self, custom_fastapi_script, resources, tmp_path_factory):
    # ... create model dir ...
    fixture_filename, target_name = custom_fastapi_script
    shutil.copy2(fixture_filename, custom_model_dir / target_name)
    return custom_model_dir
```

## Comparison with Flask Version

| Aspect | Flask | FastAPI |
|--------|-------|---------|
| Hook type | `@app.before_request` | `BaseHTTPMiddleware` |
| Response | `return jsonify(...), 401` | `return JSONResponse(..., status_code=401)` |
| Endpoint check | `request.endpoint != "model_api.ping"` | `request.url.path.rstrip("/") in ["/ping", "/"]` |
| Init function | `init_app(app)` | `init_app(app)` (same signature) |

## Notes:
- The `init_app(app: FastAPI)` function signature matches the Flask extension pattern.
- Middleware is the FastAPI equivalent of Flask's `before_request` decorator.
- Path checking uses `request.url.path` instead of Flask's `request.endpoint`.
