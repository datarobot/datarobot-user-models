# Plan: Dockerfile and Environment Changes

Updates required for base images and example environments to support FastAPI.

## Overview

To support the FastAPI migration, base Docker images must include the necessary Python packages and ensure a compatible Python version (3.8+).

## Base Image Updates

### `docker/dropin_env_base/Dockerfile`
### `docker/dropin_env_base_jdk/Dockerfile`
### `docker/dropin_env_base_r/Dockerfile`
### `docker/dropin_env_base_julia/Dockerfile`

Add FastAPI and Uvicorn dependencies:

```dockerfile
# Install FastAPI and Uvicorn
RUN pip install --no-cache-dir \
    "fastapi>=0.100.0,<1.0.0" \
    "uvicorn[standard]>=0.23.0,<1.0.0" \
    "httpx>=0.24.0,<1.0.0"
```

### Base Image requirements.txt updates

The following files also need to be updated to include FastAPI dependencies:
- `docker/dropin_env_base_jdk/requirements.txt`
- `docker/dropin_env_base_r/requirements.txt`
- `docker/dropin_env_base_julia/requirements.txt`

```
fastapi>=0.100.0,<1.0.0
uvicorn[standard]>=0.23.0,<1.0.0
httpx>=0.24.0,<1.0.0
```

## Example Environments

### `example_dropin_environments/`

Update `requirements.txt` in relevant example environments:

```
# Add to example requirements.txt if they override base dependencies
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
```

## Public Drop-in Environments Audit

There are over 20+ environments in `public_dropin_environments/` that need to be audited for FastAPI compatibility.

### Audit Checklist:
1. **Python Version**: Ensure base image uses Python 3.8+.
2. **Flask Usage**: Identify environments that use `custom_flask.py` or rely on Flask-specific behavior.
3. **Dependency Conflicts**: Check if adding `fastapi`/`uvicorn` conflicts with existing packages (e.g., old `pydantic` versions).
4. **Memory Constraints**: FastAPI/Uvicorn might have slightly higher baseline memory usage than Flask/Gunicorn/gevent in some configurations.

### Migration Path for Public Envs:
- **Phase 1**: Add FastAPI/Uvicorn to `requirements.txt` or `conda.yaml` of all Python 3.8+ environments.
- **Phase 2**: For environments using `custom_flask.py`, provide a `custom_fastapi.py` equivalent.
- **Phase 3**: Update environment smoke tests to run with `DRUM_SERVER_TYPE=fastapi`.

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
