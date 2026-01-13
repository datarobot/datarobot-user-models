# Plan: model_templates/flask_extension_httpauth Migration

Migration plan for the Flask extension example to support FastAPI.

## Overview

The `model_templates/flask_extension_httpauth/` directory contains an example of how to extend Flask with custom HTTP authentication. This example needs to be updated to support FastAPI.

## Current Files

```
model_templates/flask_extension_httpauth/
├── custom_flask.py      # Flask extension with HTTP Basic Auth
├── custom.py            # Custom model hooks
├── model-metadata.yaml  # Model metadata
├── README.md            # Documentation
├── requirements.txt     # Dependencies
└── sklearn_reg.pkl      # Pre-trained model
```

## Migration Options

### Option 1: Create Parallel FastAPI Example (Recommended)

Create a new `model_templates/fastapi_extension_httpauth/` directory:

```
model_templates/fastapi_extension_httpauth/
├── custom_fastapi.py    # FastAPI extension with HTTP Basic Auth
├── custom.py            # Custom model hooks (same as Flask version)
├── model-metadata.yaml  # Model metadata (same as Flask version)
├── README.md            # Documentation (updated for FastAPI)
├── requirements.txt     # Dependencies (updated for FastAPI)
└── sklearn_reg.pkl      # Pre-trained model (same as Flask version)
```

### Option 2: Dual-Support in Existing Template

Keep the Flask example and add FastAPI support:

```
model_templates/flask_extension_httpauth/
├── custom_flask.py      # Flask extension (existing)
├── custom_fastapi.py    # FastAPI extension (new)
├── custom.py            # Custom model hooks
├── model-metadata.yaml  # Model metadata
├── README.md            # Documentation (updated for both)
├── requirements.txt     # Dependencies (updated for both)
└── sklearn_reg.pkl      # Pre-trained model
```

## Proposed Implementation for `custom_fastapi.py`

```python
"""
FastAPI extension for HTTP Basic Authentication.
Mirrors custom_flask.py functionality.
"""
import secrets
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


security = HTTPBasic()


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Verify HTTP Basic Auth credentials.
    
    Default credentials: admin / secret
    """
    correct_username = secrets.compare_digest(credentials.username.encode("utf8"), b"admin")
    correct_password = secrets.compare_digest(credentials.password.encode("utf8"), b"secret")
    
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


class HTTPBasicAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware for HTTP Basic Auth on all endpoints except health checks.
    """
    
    SKIP_PATHS = ["/ping", "/", "/health"]
    
    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip auth for health check endpoints
        path = request.url.path.rstrip("/")
        if path in self.SKIP_PATHS or path == "":
            return await call_next(request)
        
        # Check for Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Basic "):
            return Response(
                content="Missing or invalid Authorization header",
                status_code=401,
                headers={"WWW-Authenticate": "Basic realm='DRUM'"},
            )
        
        # Decode and verify credentials
        import base64
        try:
            credentials = base64.b64decode(auth_header[6:]).decode("utf-8")
            username, password = credentials.split(":", 1)
        except Exception:
            return Response(
                content="Invalid Authorization header format",
                status_code=401,
                headers={"WWW-Authenticate": "Basic realm='DRUM'"},
            )
        
        # Verify credentials
        correct_username = secrets.compare_digest(username.encode("utf8"), b"admin")
        correct_password = secrets.compare_digest(password.encode("utf8"), b"secret")
        
        if not (correct_username and correct_password):
            return Response(
                content="Invalid credentials",
                status_code=401,
                headers={"WWW-Authenticate": "Basic realm='DRUM'"},
            )
        
        return await call_next(request)


def init_app(app: FastAPI):
    """
    Initialize the FastAPI app with HTTP Basic Auth middleware.
    
    Args:
        app: FastAPI application instance
    """
    app.add_middleware(HTTPBasicAuthMiddleware)
```

## Updated `requirements.txt`

```
# Existing dependencies
scikit-learn

# FastAPI dependencies (if not already in DRUM)
# Note: FastAPI and Starlette are typically provided by DRUM
```

## Updated `README.md`

Add a section for FastAPI:

```markdown
## FastAPI Extension

For FastAPI-based DRUM servers, use `custom_fastapi.py`:

```python
from custom_fastapi import init_app
```

The FastAPI extension uses middleware for HTTP Basic Authentication.
Endpoints `/ping/`, `/`, and `/health/` bypass authentication.

### Testing with cURL

```bash
# Without auth (should fail for /predict/)
curl http://localhost:6789/predict/ -X POST

# With auth
curl http://localhost:6789/predict/ -X POST \
  -u admin:secret \
  -F "X=@test_data.csv"
```
```

## Timeline

| Phase | Action |
|-------|--------|
| Phase 1 | Create `custom_fastapi.py` alongside `custom_flask.py` |
| Phase 2 | Update README.md with FastAPI instructions |
| Phase 3 | Test with both Flask and FastAPI servers |
| Phase 4 (after Flask removal) | Rename `custom_fastapi.py` to `custom.py` extension pattern |

## Notes:
- The dual-support approach (Option 2) is recommended during the transition period.
- After Flask is fully removed, the Flask-specific files can be deleted.
- The FastAPI security dependency (`HTTPBasic`) provides built-in support for HTTP Basic Auth.
