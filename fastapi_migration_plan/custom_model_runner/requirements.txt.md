# Plan: custom_model_runner/requirements.txt

Add new dependencies for FastAPI support.

## New Dependencies

Add the following to `requirements.txt`:

```
# FastAPI and ASGI server
fastapi>=0.100.0,<1.0.0
uvicorn[standard]>=0.23.0,<1.0.0
starlette>=0.27.0,<1.0.0  # Usually comes with fastapi, but explicit for middleware

# Async HTTP client (REQUIRED for /directAccess/ and /nim/ proxy endpoints)
httpx>=0.24.0,<1.0.0
```

## Dependency Details

| Package | Purpose | Version Constraint |
|---------|---------|-------------------|
| `fastapi` | ASGI web framework | `>=0.100.0,<1.0.0` |
| `uvicorn[standard]` | ASGI server (with uvloop, httptools) | `>=0.23.0,<1.0.0` |
| `starlette` | ASGI toolkit (middleware, routing) | `>=0.27.0,<1.0.0` |
| `httpx` | Async HTTP client for proxy endpoints | `>=0.24.0,<1.0.0` |

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
