# Plan: custom_model_runner/requirements.txt

Add new dependencies for FastAPI support.

## New Dependencies

> ⚠️ **IMPORTANT:** Use Phase 1 (Conservative) versions for initial migration to avoid Pydantic v2 conflicts.

### Phase 1: Conservative (Pydantic v1 compatible) - RECOMMENDED FOR INITIAL MIGRATION

Add the following to `requirements.txt`:

```
# FastAPI and ASGI server (Pydantic v1 compatible)
fastapi>=0.95.0,<0.100.0
uvicorn[standard]>=0.23.0,<1.0.0
starlette>=0.27.0,<0.33.0  # Compatible with FastAPI 0.99

# Pydantic v1 (explicit for ML library compatibility)
pydantic>=1.10.0,<2.0.0

# Async HTTP client (REQUIRED for /directAccess/ and /nim/ proxy endpoints)
httpx>=0.24.0,<1.0.0

# Prometheus metrics
prometheus-client>=0.17.0,<1.0.0

# OpenTelemetry instrumentation for FastAPI
opentelemetry-instrumentation-fastapi>=0.40b0

# Response comparison for testing
deepdiff>=6.0.0  # For parity testing
```

### Phase 2: Modern (Post-stabilization, Pydantic v2)

```
# FastAPI and ASGI server (Pydantic v2)
fastapi>=0.109.0,<1.0.0
uvicorn[standard]>=0.27.0,<1.0.0
starlette>=0.36.0,<1.0.0

# Pydantic v2
pydantic>=2.5.0,<3.0.0

# Async HTTP client
httpx>=0.27.0,<1.0.0

# Prometheus metrics
prometheus-client>=0.19.0,<1.0.0

# OpenTelemetry instrumentation for FastAPI
opentelemetry-instrumentation-fastapi>=0.43b0
```

## Dependency Details

| Package | Purpose | Phase 1 Version | Phase 2 Version |
|---------|---------|-----------------|-----------------|
| `fastapi` | ASGI web framework | `>=0.95.0,<0.100.0` | `>=0.109.0,<1.0.0` |
| `uvicorn[standard]` | ASGI server | `>=0.23.0,<1.0.0` | `>=0.27.0,<1.0.0` |
| `starlette` | ASGI toolkit | `>=0.27.0,<0.33.0` | `>=0.36.0,<1.0.0` |
| `pydantic` | Data validation | `>=1.10.0,<2.0.0` | `>=2.5.0,<3.0.0` |
| `httpx` | Async HTTP client | `>=0.24.0,<1.0.0` | `>=0.27.0,<1.0.0` |
| `prometheus-client` | Metrics endpoint | `>=0.17.0,<1.0.0` | `>=0.19.0,<1.0.0` |
| `opentelemetry-instrumentation-fastapi` | OTel auto-instrumentation | `>=0.40b0` | `>=0.43b0` |
| `deepdiff` | Response comparison | `>=6.0.0` | `>=6.0.0` |

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

- **FastAPI 0.100.0+**: Includes lifespan context manager support and modern middleware patterns.
- **Uvicorn 0.23.0+**: Stable release with proper signal handling and graceful shutdown.
- **Starlette 0.27.0+**: Compatible with FastAPI 0.100+ and includes BaseHTTPMiddleware.

## Uvicorn Extras

The `uvicorn[standard]` install includes:
- `uvloop` - faster event loop (Linux/macOS)
- `httptools` - faster HTTP parsing
- `websockets` - WebSocket support
- `watchfiles` - file watching for development

## Phase 2: Removing Flask Dependencies

After migration is complete (see `REMOVING_FLASK_STRATEGY.md`), remove:

```
# REMOVE after FastAPI migration:
# flask
# gunicorn
# gevent
# werkzeug
```

## Compatibility Notes

- FastAPI requires Python 3.8+
- Uvicorn requires Python 3.8+
- The `[standard]` extras for uvicorn may not work on all platforms (e.g., Windows doesn't support uvloop)
