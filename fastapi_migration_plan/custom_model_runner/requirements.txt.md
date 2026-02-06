# Plan: custom_model_runner/requirements.txt

Add new dependencies for FastAPI support with Pydantic v2 and Python 3.12.

## New Dependencies

Add the following to `requirements.txt`:

```
# FastAPI and ASGI server (Pydantic v2)
fastapi>=0.109.0,<1.0.0
uvicorn[standard]>=0.27.0,<1.0.0
starlette>=0.36.0,<1.0.0

# Pydantic v2 (required for FastAPI 0.109+)
pydantic>=2.5.0,<3.0.0

# Async HTTP client (REQUIRED for /directAccess/ and /nim/ proxy endpoints)
httpx>=0.27.0,<1.0.0

# Event loop - critical: uvloop >= 0.19.0 for Python 3.12 compatibility
uvloop>=0.19.0;platform_system!="Windows"

# Prometheus metrics
prometheus-client>=0.19.0,<1.0.0

# OpenTelemetry instrumentation for FastAPI
opentelemetry-instrumentation-fastapi>=0.43b0

# Response comparison for testing
deepdiff>=6.0.0  # For parity testing
```

## Dependency Details

| Package | Purpose | Version | Notes |
|---------|---------|---------|-------|
| `fastapi` | ASGI web framework | `>=0.109.0,<1.0.0` | Modern with Pydantic v2 |
| `uvicorn[standard]` | ASGI server | `>=0.27.0,<1.0.0` | Latest features |
| `starlette` | ASGI toolkit | `>=0.36.0,<1.0.0` | Compatible with FastAPI 0.109+ |
| `pydantic` | Data validation | `>=2.5.0,<3.0.0` | v2 for performance (5-50x faster) |
| `httpx` | Async HTTP client | `>=0.27.0,<1.0.0` | For async proxy endpoints |
| `uvloop` | Event loop | `>=0.19.0` | Python 3.12 segfault fix |
| `prometheus-client` | Metrics endpoint | `>=0.19.0,<1.0.0` | Prometheus metrics |
| `opentelemetry-instrumentation-fastapi` | OTel auto-instrumentation | `>=0.43b0` | Tracing support |
| `deepdiff` | Response comparison | `>=6.0.0` | For parity testing |

## Python Version Requirements

- **Minimum:** Python 3.9+
- **Recommended:** Python 3.11 or 3.12
- **Note:** Python 3.8 users should use `DRUM_SERVER_TYPE=flask`

## Pydantic v2 Benefits

- **Performance:** 5-50x faster validation
- **Simpler code:** No v1 compatibility layer needed
- **Future-proof:** FastAPI 1.0 will require Pydantic v2
- **Better typing:** Improved type hints and IDE support

## Known Library Compatibility

If using these libraries in custom models, ensure minimum versions:

| Library | Minimum Version | Notes |
|---------|-----------------|-------|
| `langchain` | >= 0.1.0 | Pydantic v2 support |
| `mlflow` | >= 2.9.0 | Pydantic v2 support |
| `ray[serve]` | >= 2.8.0 | Pydantic v2 support |
| `datarobot` SDK | Any | Compatible |
| `transformers` | Any | No conflict |

## Why httpx is Required

The `/directAccess/{path}` and `/nim/{path}` proxy endpoints forward requests to the NIM/OpenAI backend.
Using the synchronous `requests` library inside async FastAPI endpoints would **block the event loop**,
preventing other requests from being processed.

`httpx` provides an async HTTP client (`httpx.AsyncClient`) that integrates properly with asyncio:

```python
async with httpx.AsyncClient(timeout=timeout) as client:
    resp = await client.request(method=request.method, url=target_url, ...)
```

## Why These Versions

- **FastAPI 0.109.0+**: Modern version with full Pydantic v2 support and lifespan context manager.
- **Uvicorn 0.27.0+**: Latest stable release with proper signal handling and graceful shutdown.
- **Starlette 0.36.0+**: Compatible with FastAPI 0.109+ and includes modern middleware.
- **Pydantic 2.5.0+**: Stable v2 release with performance improvements and better typing.
- **uvloop 0.19.0+**: Critical fix for Python 3.12 segfault issues.

## Uvicorn Extras

The `uvicorn[standard]` install includes:
- `uvloop` - faster event loop (Linux/macOS) - **requires >= 0.19.0 for Python 3.12**
- `httptools` - faster HTTP parsing
- `websockets` - WebSocket support
- `watchfiles` - file watching for development

## Removing Flask Dependencies

After migration is complete (see `REMOVING_FLASK_STRATEGY.md`), remove:

```
# REMOVE after FastAPI migration:
# flask
# gunicorn
# gevent
# werkzeug
```

## Platform Notes

- **Linux/macOS:** Full uvloop support for best performance
- **Windows:** uvloop not available, falls back to asyncio (still performant)
- **Python 3.12:** Requires uvloop >= 0.19.0 to avoid segfaults
