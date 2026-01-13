# Plan: custom_model_runner/datarobot_drum/drum/fastapi/extensions.py

Mechanism to load custom FastAPI extensions, mirroring the Flask extension loading in `prediction_server.py`.

## Overview

This module provides the `load_fastapi_extensions()` function that:
1. Searches for `custom_fastapi.py` in the model directory
2. Dynamically imports the module
3. Calls `init_app(app)` to allow custom middleware, routes, or configuration

## Proposed Implementation

```python
"""
FastAPI extension loader.
Mirrors the Flask extension loading mechanism in prediction_server.py.
"""
import logging
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI

from datarobot_drum.drum.enum import FASTAPI_EXT_FILE_NAME, LOGGER_NAME_PREFIX
from datarobot_drum.drum.exceptions import DrumCommonException

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


def load_fastapi_extensions(app: FastAPI, code_dir: str) -> Optional[object]:
    """
    Load and apply custom FastAPI extensions from the model directory.
    
    This function searches for a `custom_fastapi.py` file in the model directory
    and calls its `init_app(app)` function to apply custom configurations.
    
    The extension module can:
    - Add custom middleware
    - Add custom routes/routers
    - Modify app settings
    - Add event handlers
    - Add dependencies
    
    Args:
        app: The FastAPI application instance
        code_dir: Path to the model/code directory
    
    Returns:
        The loaded module if found, None otherwise
    
    Raises:
        DrumCommonException: If the extension file is found but fails to load
        RuntimeError: If multiple extension files are found
    
    Example extension file (custom_fastapi.py):
        ```python
        from fastapi import FastAPI
        from starlette.middleware.base import BaseHTTPMiddleware
        
        class MyMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                # Custom logic
                return await call_next(request)
        
        def init_app(app: FastAPI):
            app.add_middleware(MyMiddleware)
        ```
    """
    # Search for custom_fastapi.py in the code directory
    custom_file_paths = list(Path(code_dir).rglob(f"{FASTAPI_EXT_FILE_NAME}.py"))
    
    # Check for legacy custom_flask.py and warn the user
    legacy_file_paths = list(Path(code_dir).rglob("custom_flask.py"))
    if legacy_file_paths:
        logger.warning(
            "Legacy custom_flask.py detected at %s while running with FastAPI. "
            "This file will be IGNORED. Please migrate your extensions to custom_fastapi.py "
            "referring to the USER_MIGRATION_GUIDE.md",
            legacy_file_paths
        )

    if len(custom_file_paths) > 1:
        raise RuntimeError(
            f"Found multiple custom FastAPI hook files: {custom_file_paths}. "
            "Only one custom_fastapi.py is allowed."
        )
    
    if len(custom_file_paths) == 0:
        logger.info(
            "No %s.py file detected in %s", 
            FASTAPI_EXT_FILE_NAME, 
            code_dir
        )
        return None
    
    custom_file_path = custom_file_paths[0]
    logger.info("Detected %s .. trying to load FastAPI extensions", custom_file_path)
    
    # Add the parent directory to sys.path for imports
    parent_dir = str(custom_file_path.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    try:
        # Import the module
        custom_module = __import__(FASTAPI_EXT_FILE_NAME)
        
        # Check if init_app function exists
        if not hasattr(custom_module, "init_app"):
            logger.warning(
                "%s does not have init_app() function. Skipping extension loading.",
                custom_file_path
            )
            return custom_module
        
        # Call init_app to apply extensions
        logger.info("Calling init_app() from %s", custom_file_path)
        custom_module.init_app(app)
        
        logger.info("Successfully loaded FastAPI extensions from %s", custom_file_path)
        return custom_module
        
    except ImportError as e:
        logger.error("Could not import FastAPI extension module", exc_info=True)
        raise DrumCommonException(
            f"Failed to import FastAPI extension from [{custom_file_path}]: {e}"
        )
    except Exception as e:
        logger.error("Error while loading FastAPI extensions", exc_info=True)
        raise DrumCommonException(
            f"Failed to apply FastAPI extensions from [{custom_file_path}]: {e}"
        )


def validate_extension_module(module) -> bool:
    """
    Validate that an extension module has the required interface.
    
    Args:
        module: The imported module to validate
    
    Returns:
        True if the module is valid, False otherwise
    """
    if not hasattr(module, "init_app"):
        logger.warning("Extension module missing init_app() function")
        return False
    
    if not callable(module.init_app):
        logger.warning("Extension module init_app is not callable")
        return False
    
    return True
```

## Extension Module Interface

Custom FastAPI extensions must implement the following interface:

```python
# custom_fastapi.py

from fastapi import FastAPI

def init_app(app: FastAPI) -> None:
    """
    Initialize the FastAPI application with custom extensions.
    
    This function is called by DRUM after the base FastAPI app is created
    but before the server starts.
    
    Args:
        app: The FastAPI application instance
    """
    # Add middleware
    # app.add_middleware(MyMiddleware)
    
    # Add routes
    # app.include_router(my_router)
    
    # Add event handlers
    # @app.on_event("startup")
    # async def startup():
    #     pass
    
    pass
```

## Supported Extension Types

| Extension Type | How to Add |
|---------------|------------|
| Middleware | `app.add_middleware(MyMiddleware)` |
| Routes | `app.include_router(router)` or `@app.get("/path")` |
| Exception Handlers | `@app.exception_handler(Exception)` |
| Dependencies | `app.dependency_overrides[dep] = override` |
| Event Handlers | `@app.on_event("startup")` / `@app.on_event("shutdown")` |

## Example Extensions

### Authentication Middleware

```python
# custom_fastapi.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class AuthMiddleware(BaseHTTPMiddleware):
    SKIP_PATHS = ["/ping", "/", "/health"]
    
    async def dispatch(self, request: Request, call_next):
        if request.url.path.rstrip("/") in self.SKIP_PATHS:
            return await call_next(request)
        
        auth_token = request.headers.get("Authorization")
        if not auth_token:
            return JSONResponse(
                content={"message": "Missing Authorization header"},
                status_code=401
            )
        
        return await call_next(request)


def init_app(app: FastAPI):
    app.add_middleware(AuthMiddleware)
```

### Custom Routes

```python
# custom_fastapi.py
from fastapi import FastAPI, APIRouter

router = APIRouter()

@router.get("/custom/status")
async def custom_status():
    return {"status": "custom endpoint working"}


def init_app(app: FastAPI):
    app.include_router(router)
```

### Request Logging

```python
# custom_fastapi.py
import logging
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        logger.info(f"Request: {request.method} {request.url.path}")
        response = await call_next(request)
        logger.info(f"Response: {response.status_code}")
        return response


def init_app(app: FastAPI):
    app.add_middleware(RequestLoggingMiddleware)
```

## Comparison with Flask Extension Loading

| Aspect | Flask | FastAPI |
|--------|-------|---------|
| File name | `custom_flask.py` | `custom_fastapi.py` |
| Function | `init_app(app)` | `init_app(app)` |
| Middleware | `@app.before_request` | `BaseHTTPMiddleware` |
| Routes | `Blueprint` | `APIRouter` |
| Constant | `FLASK_EXT_FILE_NAME` | `FASTAPI_EXT_FILE_NAME` |

## Notes

- Only one `custom_fastapi.py` file is allowed per model directory.
- The extension is loaded after the base FastAPI app is created.
- Extensions can access the full FastAPI application instance.
- Errors during extension loading are logged and raised as `DrumCommonException`.
- The parent directory of the extension file is added to `sys.path` for imports.
