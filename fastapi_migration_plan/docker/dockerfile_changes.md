# Plan: Dockerfile and Environment Changes

Updates required for base images and example environments to support FastAPI.

## Overview

To support the FastAPI migration, base Docker images must include the necessary Python packages and ensure a compatible Python version (3.8+).

## Base Image Updates

### `docker/dropin_env_base/Dockerfile`

Add FastAPI and Uvicorn dependencies:

```dockerfile
# Install FastAPI and Uvicorn
RUN pip install --no-cache-dir \
    "fastapi>=0.100.0,<1.0.0" \
    "uvicorn[standard]>=0.23.0,<1.0.0" \
    "httpx>=0.24.0,<1.0.0"
```

### `docker/dropin_env_base_jdk/Dockerfile`
### `docker/dropin_env_base_r/Dockerfile`

Similar changes are required for all base images that provide a Python runtime for DRUM.

## Example Environments

### `example_dropin_environments/`

Update `requirements.txt` in relevant example environments:

```
# Add to example requirements.txt if they override base dependencies
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
```

## Python Version Requirements

FastAPI and its dependencies require Python 3.8 or higher. We must audit all base images and ensure they are using a compatible version.

| Image | Current Python | Action |
|-------|----------------|--------|
| `dropin_env_base` | 3.7? | Upgrade to 3.8+ if needed |
| `dropin_env_base_jdk` | 3.7? | Upgrade to 3.8+ if needed |
| `dropin_env_base_r` | 3.7? | Upgrade to 3.8+ if needed |

## Coexistence Strategy

During the transition period, both Flask and FastAPI dependencies will be present in the base images. This ensures:
1. Existing Flask-based models continue to work.
2. New FastAPI-based models can be tested and deployed.
3. Functional tests can run against both server types.

## Cleanup (Phase 5)

Once the FastAPI migration is confirmed stable, we will:
1. Remove `flask`, `gunicorn`, and `gevent` from base images.
2. Update all examples to exclusively use FastAPI.
3. Remove backward compatibility checks from `entry_point.py`.

## Notes:
- The `uvicorn[standard]` extra is important for performance (includes `uvloop` and `httptools`).
- `httpx` is required for async proxying in the `/directAccess/` and `/nim/` endpoints.
