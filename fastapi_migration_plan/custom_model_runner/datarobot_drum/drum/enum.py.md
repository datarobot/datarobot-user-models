# Plan: custom_model_runner/datarobot_drum/drum/enum.py

Add FastAPI extension file name constant and other required constants.

## Changes:
- Add `FASTAPI_EXT_FILE_NAME = "custom_fastapi"` (parallel to `FLASK_EXT_FILE_NAME = "custom_flask"`).
- Add `DRUM_FASTAPI_ENABLED = "DRUM_FASTAPI_ENABLED"` for feature flag control.
- Add `DRUM_SERVER_TYPE = "DRUM_SERVER_TYPE"` for server selection.
- Add SSL/TLS configuration constants.
- Add FastAPI performance tuning constants.

```python
# Existing:
FLASK_EXT_FILE_NAME = "custom_flask"

# Add new:
FASTAPI_EXT_FILE_NAME = "custom_fastapi"

# Feature flag and selection
DRUM_FASTAPI_ENABLED = "DRUM_FASTAPI_ENABLED"
DRUM_SERVER_TYPE = "DRUM_SERVER_TYPE"

# FastAPI / Uvicorn Tuning
DRUM_FASTAPI_EXECUTOR_WORKERS = "DRUM_FASTAPI_EXECUTOR_WORKERS"
DRUM_FASTAPI_MAX_UPLOAD_SIZE = "DRUM_FASTAPI_MAX_UPLOAD_SIZE"
DRUM_FASTAPI_ENABLE_DOCS = "DRUM_FASTAPI_ENABLE_DOCS"
DRUM_UVICORN_LOOP = "DRUM_UVICORN_LOOP"
DRUM_UVICORN_MAX_REQUESTS = "DRUM_UVICORN_MAX_REQUESTS"
DRUM_UVICORN_GRACEFUL_TIMEOUT = "DRUM_UVICORN_GRACEFUL_TIMEOUT"
DRUM_UVICORN_KEEP_ALIVE = "DRUM_UVICORN_KEEP_ALIVE"
DRUM_UVICORN_LOG_LEVEL = "DRUM_UVICORN_LOG_LEVEL"

# SSL/TLS Configuration
DRUM_SSL_CERTFILE = "DRUM_SSL_CERTFILE"
DRUM_SSL_KEYFILE = "DRUM_SSL_KEYFILE"
DRUM_SSL_KEYFILE_PASSWORD = "DRUM_SSL_KEYFILE_PASSWORD"
DRUM_SSL_VERSION = "DRUM_SSL_VERSION"
DRUM_SSL_CERT_REQS = "DRUM_SSL_CERT_REQS"
DRUM_SSL_CA_CERTS = "DRUM_SSL_CA_CERTS"
DRUM_SSL_CIPHERS = "DRUM_SSL_CIPHERS"
```

This constant will be used by `load_fastapi_extensions()` in `fastapi/extensions.py` to locate custom FastAPI hook files and by `config.py` for server tuning.
