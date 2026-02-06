# Plan: custom_model_runner/datarobot_drum/drum/fastapi/config.py

Configuration handler for Uvicorn parameters, mirroring `gunicorn.conf.py`.

## Overview

This module extracts runtime parameters and maps them to Uvicorn configuration. It provides a unified configuration interface for the Uvicorn server.

## Important: Timeout Parameter Mapping

**WARNING:** `timeout_keep_alive` in Uvicorn is NOT the same as `timeout` in Gunicorn!

| Parameter | Gunicorn | Uvicorn | Purpose |
|-----------|----------|---------|---------|
| Request timeout | `timeout` (worker killed after N seconds) | **Custom middleware** | Max time for request processing |
| HTTP keep-alive | `keepalive` | `timeout_keep_alive` | Time to wait for next request on keep-alive connection |

Since Uvicorn doesn't have a built-in request timeout, we implement it via `RequestTimeoutMiddleware`.

## Proposed Implementation:

```python
"""
Uvicorn configuration module.
Maps DRUM RuntimeParameters to Uvicorn settings.
"""
import logging
import os
import platform
import ssl
import warnings
from dataclasses import dataclass, field
from typing import Optional, List

from datarobot_drum import RuntimeParameters
from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)

# Secure SSL ciphers for TLS 1.2+
# Excludes weak ciphers (MD5, RC4, 3DES, etc.)
SECURE_SSL_CIPHERS = (
    "ECDHE+AESGCM:"
    "DHE+AESGCM:"
    "ECDHE+CHACHA20:"
    "DHE+CHACHA20:"
    "!aNULL:!MD5:!DSS:!RC4:!3DES"
)


def _is_uvloop_available() -> bool:
    """Check if uvloop is available on this platform."""
    if platform.system() == "Windows":
        return False
    try:
        import uvloop  # noqa: F401
        return True
    except ImportError:
        return False


def _is_uvloop_python312_compatible() -> bool:
    """
    Check if uvloop version is compatible with Python 3.12.
    
    Python 3.12 requires uvloop >= 0.19.0 to avoid segfaults.
    See: https://github.com/MagicStack/uvloop/issues/513
    """
    import sys
    
    if sys.version_info < (3, 12):
        return True  # Not Python 3.12, no special check needed
    
    try:
        import uvloop
        version_parts = uvloop.__version__.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
        
        # uvloop >= 0.19.0 required for Python 3.12
        return (major, minor) >= (0, 19)
    except (ImportError, ValueError, AttributeError):
        return False


@dataclass
class UvicornConfig:
    """
    Configuration for Uvicorn server.
    
    This class separates:
    - Uvicorn server settings (host, port, workers, etc.)
    - Application-level settings (request_timeout, max_upload_size, etc.)
    
    Application-level settings are handled by middleware, not Uvicorn itself.
    """
    # Uvicorn server settings
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    timeout_keep_alive: int = 5  # HTTP keep-alive, NOT request timeout!
    timeout_graceful_shutdown: Optional[int] = None
    timeout_notify: int = 30  # Time to wait for headers (slow loris protection)
    backlog: int = 2048
    log_level: str = "info"
    loop: str = "auto"
    limit_max_requests: Optional[int] = None
    limit_concurrency: Optional[int] = None  # Max concurrent connections
    
    # Application-level settings (used by middleware, NOT Uvicorn)
    executor_workers: int = 4
    max_upload_size: int = 100 * 1024 * 1024  # 100MB
    request_timeout: float = 120.0  # Request timeout in seconds (middleware)
    
    # SSL/TLS Configuration
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    ssl_keyfile_password: Optional[str] = None
    ssl_version: int = field(default_factory=lambda: ssl.PROTOCOL_TLS_SERVER)
    ssl_cert_reqs: int = field(default_factory=lambda: ssl.CERT_NONE)
    ssl_ca_certs: Optional[str] = None
    ssl_ciphers: str = SECURE_SSL_CIPHERS
    
    @classmethod
    def from_runtime_params(cls) -> "UvicornConfig":
        """Create config from RuntimeParameters and environment variables."""
        config = cls()
        
        # Parse ADDRESS environment variable (format: "host:port")
        address = os.environ.get("ADDRESS", "0.0.0.0:8080")
        if ":" in address:
            config.host, port_str = address.rsplit(":", 1)
            config.port = int(port_str)
        else:
            config.host = address
        
        # Workers (equivalent to CUSTOM_MODEL_WORKERS / MAX_WORKERS in gunicorn)
        if RuntimeParameters.has("CUSTOM_MODEL_WORKERS"):
            temp_workers = int(RuntimeParameters.get("CUSTOM_MODEL_WORKERS"))
            if 0 < temp_workers < 200:
                config.workers = temp_workers
        elif os.environ.get("MAX_WORKERS"):
            temp_workers = int(os.environ.get("MAX_WORKERS"))
            if 0 < temp_workers < 200:
                config.workers = temp_workers
        
        # Backlog (equivalent to DRUM_WEBSERVER_BACKLOG)
        if RuntimeParameters.has("DRUM_WEBSERVER_BACKLOG"):
            temp_backlog = int(RuntimeParameters.get("DRUM_WEBSERVER_BACKLOG"))
            if 1 <= temp_backlog <= 10000:
                config.backlog = temp_backlog
        
        # REQUEST TIMEOUT (for middleware, NOT Uvicorn's timeout_keep_alive!)
        # This is the equivalent of Gunicorn's --timeout
        if RuntimeParameters.has("DRUM_CLIENT_REQUEST_TIMEOUT"):
            temp_timeout = float(RuntimeParameters.get("DRUM_CLIENT_REQUEST_TIMEOUT"))
            if 0 < temp_timeout <= 3600:
                config.request_timeout = temp_timeout
            logger.info(
                "Request timeout set to %.1fs (enforced by RequestTimeoutMiddleware)",
                config.request_timeout
            )
        
        # HTTP Keep-Alive (NOT request timeout!)
        # This controls how long to wait for the next request on a keep-alive connection
        if RuntimeParameters.has("DRUM_UVICORN_KEEP_ALIVE"):
            temp_keepalive = int(RuntimeParameters.get("DRUM_UVICORN_KEEP_ALIVE"))
            if 1 <= temp_keepalive <= 3600:
                config.timeout_keep_alive = temp_keepalive
        elif RuntimeParameters.has("DRUM_GUNICORN_KEEP_ALIVE"):
            temp_keepalive = int(RuntimeParameters.get("DRUM_GUNICORN_KEEP_ALIVE"))
            if 1 <= temp_keepalive <= 3600:
                config.timeout_keep_alive = temp_keepalive
                warnings.warn(
                    "DRUM_GUNICORN_KEEP_ALIVE is deprecated for FastAPI server. "
                    "Use DRUM_UVICORN_KEEP_ALIVE instead.",
                    DeprecationWarning
                )
        
        # Max requests before worker restart (memory leak mitigation)
        if RuntimeParameters.has("DRUM_UVICORN_MAX_REQUESTS"):
            temp_max_requests = int(RuntimeParameters.get("DRUM_UVICORN_MAX_REQUESTS"))
            if 0 < temp_max_requests <= 100000:
                config.limit_max_requests = temp_max_requests
        elif RuntimeParameters.has("DRUM_GUNICORN_MAX_REQUESTS"):
            temp_max_requests = int(RuntimeParameters.get("DRUM_GUNICORN_MAX_REQUESTS"))
            if 0 < temp_max_requests <= 100000:
                config.limit_max_requests = temp_max_requests
                warnings.warn(
                    "DRUM_GUNICORN_MAX_REQUESTS is deprecated for FastAPI server. "
                    "Use DRUM_UVICORN_MAX_REQUESTS instead.",
                    DeprecationWarning
                )
        
        # Graceful shutdown timeout
        if RuntimeParameters.has("DRUM_UVICORN_GRACEFUL_TIMEOUT"):
            temp_graceful = int(RuntimeParameters.get("DRUM_UVICORN_GRACEFUL_TIMEOUT"))
            if 1 <= temp_graceful <= 3600:
                config.timeout_graceful_shutdown = temp_graceful
        elif RuntimeParameters.has("DRUM_GUNICORN_GRACEFUL_TIMEOUT"):
            temp_graceful = int(RuntimeParameters.get("DRUM_GUNICORN_GRACEFUL_TIMEOUT"))
            if 1 <= temp_graceful <= 3600:
                config.timeout_graceful_shutdown = temp_graceful
                warnings.warn(
                    "DRUM_GUNICORN_GRACEFUL_TIMEOUT is deprecated for FastAPI server. "
                    "Use DRUM_UVICORN_GRACEFUL_TIMEOUT instead.",
                    DeprecationWarning
                )
        
        # Log level
        if RuntimeParameters.has("DRUM_UVICORN_LOG_LEVEL"):
            temp_loglevel = str(RuntimeParameters.get("DRUM_UVICORN_LOG_LEVEL")).lower()
            valid_levels = {"debug", "info", "warning", "error", "critical", "trace"}
            if temp_loglevel in valid_levels:
                config.log_level = temp_loglevel
        elif RuntimeParameters.has("DRUM_GUNICORN_LOG_LEVEL"):
            temp_loglevel = str(RuntimeParameters.get("DRUM_GUNICORN_LOG_LEVEL")).lower()
            valid_levels = {"debug", "info", "warning", "error", "critical"}
            if temp_loglevel in valid_levels:
                config.log_level = temp_loglevel
        
        # Event loop implementation with platform detection
        config.loop = cls._determine_event_loop()
        
        # Executor workers for sync predictions
        if RuntimeParameters.has("DRUM_FASTAPI_EXECUTOR_WORKERS"):
            config.executor_workers = int(RuntimeParameters.get("DRUM_FASTAPI_EXECUTOR_WORKERS"))
        
        # Max upload size
        if RuntimeParameters.has("DRUM_FASTAPI_MAX_UPLOAD_SIZE"):
            config.max_upload_size = int(RuntimeParameters.get("DRUM_FASTAPI_MAX_UPLOAD_SIZE"))
        
        # Concurrent connection limit
        if RuntimeParameters.has("DRUM_UVICORN_LIMIT_CONCURRENCY"):
            config.limit_concurrency = int(RuntimeParameters.get("DRUM_UVICORN_LIMIT_CONCURRENCY"))
        
        # Header timeout (slow loris protection)
        if RuntimeParameters.has("DRUM_UVICORN_HEADER_TIMEOUT"):
            config.timeout_notify = int(RuntimeParameters.get("DRUM_UVICORN_HEADER_TIMEOUT"))

        # SSL Configuration
        config._load_ssl_config()
        
        return config
    
    @staticmethod
    def _determine_event_loop() -> str:
        """
        Determine the best event loop for this platform.
        
        Priority:
        1. Explicit DRUM_UVICORN_LOOP setting (with validation)
        2. uvloop on Linux/macOS if available and compatible
        3. asyncio as fallback
        
        Note: Python 3.12 requires uvloop >= 0.19.0 to avoid segfaults.
        """
        import sys
        system = platform.system()
        
        if RuntimeParameters.has("DRUM_UVICORN_LOOP"):
            requested = str(RuntimeParameters.get("DRUM_UVICORN_LOOP")).lower()
            
            if requested == "uvloop":
                if system == "Windows":
                    logger.warning(
                        "uvloop is not available on Windows. "
                        "Falling back to asyncio event loop."
                    )
                    return "asyncio"
                elif not _is_uvloop_available():
                    logger.warning(
                        "uvloop requested but not installed. "
                        "Install with: pip install uvloop. "
                        "Falling back to asyncio."
                    )
                    return "asyncio"
                elif not _is_uvloop_python312_compatible():
                    # Python 3.12 requires uvloop >= 0.19.0
                    try:
                        import uvloop
                        logger.warning(
                            "uvloop %s may cause segfaults on Python 3.12. "
                            "Upgrade to uvloop>=0.19.0 or falling back to asyncio. "
                            "Install with: pip install 'uvloop>=0.19.0'",
                            uvloop.__version__
                        )
                    except ImportError:
                        pass
                    return "asyncio"
                return "uvloop"
            
            if requested in {"auto", "asyncio"}:
                return requested
            
            logger.warning(
                "Invalid DRUM_UVICORN_LOOP value: %s. "
                "Valid options: auto, asyncio, uvloop. Using 'auto'.",
                requested
            )
            return "auto"
        
        # Default: auto (uvicorn will pick uvloop if available on Linux/macOS)
        # But check Python 3.12 compatibility first
        if sys.version_info >= (3, 12) and _is_uvloop_available() and not _is_uvloop_python312_compatible():
            try:
                import uvloop
                logger.warning(
                    "Python 3.12 detected with uvloop %s. "
                    "uvloop < 0.19.0 may cause segfaults. "
                    "Using asyncio instead. Upgrade with: pip install 'uvloop>=0.19.0'",
                    uvloop.__version__
                )
            except ImportError:
                pass
            return "asyncio"
        
        return "auto"
    
    def _load_ssl_config(self):
        """Load SSL configuration from runtime parameters."""
        if RuntimeParameters.has("DRUM_SSL_CERTFILE"):
            self.ssl_certfile = str(RuntimeParameters.get("DRUM_SSL_CERTFILE"))
            
            # Validate certificate file exists
            if not os.path.isfile(self.ssl_certfile):
                raise ValueError(f"SSL certificate file not found: {self.ssl_certfile}")
        
        if RuntimeParameters.has("DRUM_SSL_KEYFILE"):
            self.ssl_keyfile = str(RuntimeParameters.get("DRUM_SSL_KEYFILE"))
            
            if not os.path.isfile(self.ssl_keyfile):
                raise ValueError(f"SSL key file not found: {self.ssl_keyfile}")
        
        if RuntimeParameters.has("DRUM_SSL_KEYFILE_PASSWORD"):
            self.ssl_keyfile_password = str(RuntimeParameters.get("DRUM_SSL_KEYFILE_PASSWORD"))
        
        if RuntimeParameters.has("DRUM_SSL_VERSION"):
            self.ssl_version = int(RuntimeParameters.get("DRUM_SSL_VERSION"))
        
        if RuntimeParameters.has("DRUM_SSL_CERT_REQS"):
            self.ssl_cert_reqs = int(RuntimeParameters.get("DRUM_SSL_CERT_REQS"))
        
        if RuntimeParameters.has("DRUM_SSL_CA_CERTS"):
            self.ssl_ca_certs = str(RuntimeParameters.get("DRUM_SSL_CA_CERTS"))
            
            if not os.path.isfile(self.ssl_ca_certs):
                raise ValueError(f"SSL CA certificates file not found: {self.ssl_ca_certs}")
        
        if RuntimeParameters.has("DRUM_SSL_CIPHERS"):
            self.ssl_ciphers = str(RuntimeParameters.get("DRUM_SSL_CIPHERS"))
        
        # Log SSL status
        if self.ssl_certfile:
            logger.info(
                "SSL enabled: cert=%s, key=%s, ciphers=%s",
                self.ssl_certfile,
                self.ssl_keyfile,
                self.ssl_ciphers[:50] + "..." if len(self.ssl_ciphers) > 50 else self.ssl_ciphers
            )
    
    def to_uvicorn_kwargs(self) -> dict:
        """
        Convert config to kwargs for uvicorn.run().
        
        Note: request_timeout is NOT included here - it's handled by middleware.
        """
        kwargs = {
            "host": self.host,
            "port": self.port,
            "workers": self.workers,
            "timeout_keep_alive": self.timeout_keep_alive,
            "timeout_notify": self.timeout_notify,
            "backlog": self.backlog,
            "log_level": self.log_level,
            "loop": self.loop,
            "access_log": True,
        }
        
        if self.timeout_graceful_shutdown is not None:
            kwargs["timeout_graceful_shutdown"] = self.timeout_graceful_shutdown
        
        if self.limit_max_requests is not None and self.limit_max_requests > 0:
            kwargs["limit_max_requests"] = self.limit_max_requests
        
        if self.limit_concurrency is not None and self.limit_concurrency > 0:
            kwargs["limit_concurrency"] = self.limit_concurrency

        # SSL support
        if self.ssl_certfile:
            kwargs["ssl_certfile"] = self.ssl_certfile
            kwargs["ssl_keyfile"] = self.ssl_keyfile
            if self.ssl_keyfile_password:
                kwargs["ssl_keyfile_password"] = self.ssl_keyfile_password
            kwargs["ssl_version"] = self.ssl_version
            kwargs["ssl_cert_reqs"] = self.ssl_cert_reqs
            if self.ssl_ca_certs:
                kwargs["ssl_ca_certs"] = self.ssl_ca_certs
            kwargs["ssl_ciphers"] = self.ssl_ciphers
        
        return kwargs
    
    def to_cli_args(self) -> List[str]:
        """Convert config to Uvicorn CLI arguments."""
        args = [
            "--host", self.host,
            "--port", str(self.port),
            "--workers", str(self.workers),
            "--timeout-keep-alive", str(self.timeout_keep_alive),
            "--timeout-notify", str(self.timeout_notify),
            "--backlog", str(self.backlog),
            "--log-level", self.log_level,
            "--loop", self.loop,
        ]
        
        if self.timeout_graceful_shutdown is not None:
            args.extend(["--timeout-graceful-shutdown", str(self.timeout_graceful_shutdown)])
        
        if self.limit_max_requests is not None and self.limit_max_requests > 0:
            args.extend(["--limit-max-requests", str(self.limit_max_requests)])
        
        if self.limit_concurrency is not None and self.limit_concurrency > 0:
            args.extend(["--limit-concurrency", str(self.limit_concurrency)])

        # SSL support in CLI
        if self.ssl_certfile:
            args.extend(["--ssl-certfile", self.ssl_certfile])
            if self.ssl_keyfile:
                args.extend(["--ssl-keyfile", self.ssl_keyfile])
            if self.ssl_keyfile_password:
                args.extend(["--ssl-keyfile-password", self.ssl_keyfile_password])
            args.extend(["--ssl-version", str(self.ssl_version)])
            args.extend(["--ssl-cert-reqs", str(self.ssl_cert_reqs)])
            if self.ssl_ca_certs:
                args.extend(["--ssl-ca-certs", self.ssl_ca_certs])
            args.extend(["--ssl-ciphers", self.ssl_ciphers])
        
        return args
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of warnings.
        
        Returns:
            List of warning messages (empty if all OK).
        """
        warnings_list = []
        
        # Check workers vs executor_workers ratio
        if self.workers > 1 and self.executor_workers < self.workers:
            warnings_list.append(
                f"executor_workers ({self.executor_workers}) is less than workers "
                f"({self.workers}). Consider increasing executor_workers for better "
                f"parallelism of sync model predictions."
            )
        
        # Check request timeout vs keep-alive
        if self.request_timeout < self.timeout_keep_alive:
            warnings_list.append(
                f"request_timeout ({self.request_timeout}s) is less than "
                f"timeout_keep_alive ({self.timeout_keep_alive}s). This may cause "
                f"connections to be dropped unexpectedly."
            )
        
        # Check SSL configuration
        if self.ssl_certfile and not self.ssl_keyfile:
            warnings_list.append(
                "SSL certificate provided but no key file. Server may fail to start."
            )
        
        return warnings_list
```

## Corrected Parameter Mapping

| DRUM Parameter | Purpose | Gunicorn | Uvicorn | Notes |
|----------------|---------|----------|---------|-------|
| `DRUM_CLIENT_REQUEST_TIMEOUT` | Request processing timeout | `timeout` | **`config.request_timeout`** → Middleware | NOT `timeout_keep_alive`! |
| `DRUM_UVICORN_KEEP_ALIVE` | HTTP keep-alive | `keepalive` | `timeout_keep_alive` | Time to wait for next request |
| `CUSTOM_MODEL_WORKERS` | Process count | `workers` | `workers` | |
| `DRUM_WEBSERVER_BACKLOG` | Connection queue | `backlog` | `backlog` | |
| `DRUM_UVICORN_MAX_REQUESTS` | Worker recycling | `max_requests` | `limit_max_requests` | Memory leak protection |
| `DRUM_UVICORN_GRACEFUL_TIMEOUT` | Shutdown timeout | `graceful_timeout` | `timeout_graceful_shutdown` | |
| `DRUM_UVICORN_LOG_LEVEL` | Logging level | `loglevel` | `log_level` | |
| `DRUM_UVICORN_LOOP` | Event loop | N/A (gevent) | `loop` | Windows: asyncio only |
| `DRUM_FASTAPI_EXECUTOR_WORKERS` | Thread pool size | N/A | App-level | For sync prediction |
| `DRUM_FASTAPI_EXECUTOR_QUEUE_DEPTH` | Max pending requests | N/A | App-level | Backpressure |
| `DRUM_FASTAPI_EXECUTOR_QUEUE_TIMEOUT` | Queue wait timeout | N/A | App-level | 503 on timeout |
| `DRUM_FASTAPI_MAX_UPLOAD_SIZE` | Request body limit | N/A | App-level | Middleware |

## Timeout Behavior Differences: Gunicorn vs Uvicorn

> ⚠️ **IMPORTANT:** Gunicorn and Uvicorn handle timeouts fundamentally differently!

| Aspect | Gunicorn | Uvicorn + Middleware |
|--------|----------|---------------------|
| **Mechanism** | Worker killed by arbiter | Request cancelled by middleware |
| **Resource cleanup** | ✅ Guaranteed (new process) | ⚠️ May leak if sync code ignores cancellation |
| **Memory leaks** | ✅ Prevented by process restart | ⚠️ Need `limit_max_requests` as backup |
| **Graceful handling** | ❌ Abrupt termination | ✅ Can return proper error response |

### Achieving Gunicorn-like Behavior in Uvicorn

To get similar reliability to Gunicorn's timeout, we implement **multiple layers of protection**:

```python
# Layer 1: Request timeout middleware (graceful)
# - Cancels async request after N seconds
# - Returns 504 Gateway Timeout to client
# - Does NOT kill worker

# Layer 2: Worker recycling (memory leak protection)
# - limit_max_requests: restart worker after N requests
# - Prevents memory leaks from accumulating

# Layer 3: Memory monitoring (optional, external)
# - Monitor /stats/ endpoint for RSS growth
# - Restart containers if memory exceeds threshold
```

### Memory Leak Protection Configuration

```python
# In config.py - recommended production settings

@dataclass
class UvicornConfig:
    # ...existing fields...
    
    # Worker recycling for memory leak protection
    limit_max_requests: Optional[int] = 10000  # Restart after 10K requests
    limit_max_requests_jitter: int = 1000       # Add randomness to prevent thundering herd
    
    @classmethod
    def from_runtime_params(cls) -> "UvicornConfig":
        # ...existing code...
        
        # Memory leak protection via worker recycling
        # This is the closest equivalent to Gunicorn's timeout killing workers
        if not RuntimeParameters.has("DRUM_UVICORN_MAX_REQUESTS"):
            # Default: enable worker recycling for memory protection
            config.limit_max_requests = 10000
            logger.info(
                "Worker recycling enabled: max_requests=%d (memory leak protection). "
                "Disable with DRUM_UVICORN_MAX_REQUESTS=0",
                config.limit_max_requests
            )
```

### Request Timeout Middleware Implementation

```python
# In middleware.py

import asyncio
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    """
    Middleware that enforces request timeout.
    
    IMPORTANT: This is NOT equivalent to Gunicorn's timeout!
    - Gunicorn timeout: kills the worker process (guaranteed cleanup)
    - This middleware: cancels the asyncio task (may not stop sync code)
    
    For sync prediction code running in ThreadPoolExecutor:
    - The thread continues running even after timeout
    - The response is sent to client, but thread may leak
    - Use limit_max_requests to periodically restart workers
    """
    
    def __init__(self, app, timeout_seconds: float = 120.0):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds
    
    async def dispatch(self, request: Request, call_next):
        # Skip timeout for health endpoints (they should be fast)
        if request.url.path.rstrip("/") in ["/ping", "/health", "/livez", "/readyz", "/stats"]:
            return await call_next(request)
        
        try:
            return await asyncio.wait_for(
                call_next(request),
                timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.error(
                "Request timeout after %.1fs: %s %s",
                self.timeout_seconds,
                request.method,
                request.url.path
            )
            return JSONResponse(
                status_code=504,
                content={
                    "message": f"Request timed out after {self.timeout_seconds}s",
                    "error": "GATEWAY_TIMEOUT"
                }
            )


class MemoryGuardMiddleware(BaseHTTPMiddleware):
    """
    Middleware that monitors memory usage and triggers worker restart.
    
    This provides an additional layer of protection similar to Gunicorn's
    worker restart on timeout.
    """
    
    def __init__(
        self, 
        app, 
        max_memory_mb: int = 2048,
        check_interval_requests: int = 100
    ):
        super().__init__(app)
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.check_interval = check_interval_requests
        self._request_count = 0
        self._should_restart = False
    
    async def dispatch(self, request: Request, call_next):
        self._request_count += 1
        
        # Periodic memory check
        if self._request_count % self.check_interval == 0:
            import psutil
            process = psutil.Process()
            rss = process.memory_info().rss
            
            if rss > self.max_memory_bytes:
                logger.warning(
                    "Memory usage %.1f MB exceeds limit %.1f MB. "
                    "Worker will restart after current requests complete.",
                    rss / 1024 / 1024,
                    self.max_memory_bytes / 1024 / 1024
                )
                self._should_restart = True
        
        response = await call_next(request)
        
        # Signal uvicorn to restart this worker
        if self._should_restart:
            response.headers["X-Worker-Should-Restart"] = "true"
            # Uvicorn will restart worker when limit_max_requests is reached
            # We can accelerate this by setting a flag
        
        return response
```

## Platform-Specific Event Loop

| Platform | uvloop available | Default loop |
|----------|------------------|--------------|
| Linux x86_64 | ✅ | auto (uvloop) |
| Linux ARM64 | ✅ | auto (uvloop) |
| macOS Intel | ✅ | auto (uvloop) |
| macOS Apple Silicon | ✅ | auto (uvloop) |
| **Windows** | ❌ | asyncio |
| Alpine Linux | ⚠️ | auto (if built) |

## SSL/TLS Best Practices

### Recommended Cipher Suites (2024+)

The default `SECURE_SSL_CIPHERS` string excludes:
- MD5 (weak hash)
- RC4 (broken)
- 3DES (slow, weak)
- DSS (deprecated)
- aNULL (no authentication)

### Testing SSL Configuration

```bash
# Generate self-signed cert for testing
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes \
  -subj "/CN=localhost"

# Start server with SSL
export MLOPS_RUNTIME_PARAM_DRUM_SSL_CERTFILE=cert.pem
export MLOPS_RUNTIME_PARAM_DRUM_SSL_KEYFILE=key.pem
export MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE=fastapi
drum server --code-dir ./model --address 0.0.0.0:8443

# Test connection
curl -k https://localhost:8443/ping

# Verify TLS version and ciphers
openssl s_client -connect localhost:8443 -tls1_2 </dev/null 2>/dev/null | grep -E "Protocol|Cipher"
```

## Deprecation Warnings

The following parameters are deprecated for FastAPI and will emit warnings:

| Deprecated | Replacement |
|------------|-------------|
| `DRUM_GUNICORN_KEEP_ALIVE` | `DRUM_UVICORN_KEEP_ALIVE` |
| `DRUM_GUNICORN_MAX_REQUESTS` | `DRUM_UVICORN_MAX_REQUESTS` |
| `DRUM_GUNICORN_GRACEFUL_TIMEOUT` | `DRUM_UVICORN_GRACEFUL_TIMEOUT` |

These will continue to work but log deprecation warnings.

## Unit Tests

```python
# tests/unit/datarobot_drum/drum/fastapi/test_config.py
import pytest
import platform
from unittest.mock import patch, MagicMock

class TestUvicornConfig:
    def test_request_timeout_not_keep_alive(self):
        """Ensure request timeout is separate from keep-alive."""
        with patch.object(RuntimeParameters, "has", return_value=True):
            with patch.object(RuntimeParameters, "get", return_value="300"):
                config = UvicornConfig.from_runtime_params()
        
        # request_timeout should be set
        assert config.request_timeout == 300.0
        # timeout_keep_alive should NOT be affected
        assert config.timeout_keep_alive == 5  # default
    
    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows only")
    def test_uvloop_fallback_on_windows(self):
        """uvloop should fall back to asyncio on Windows."""
        with patch.object(RuntimeParameters, "has", return_value=True):
            with patch.object(RuntimeParameters, "get", return_value="uvloop"):
                config = UvicornConfig.from_runtime_params()
        
        assert config.loop == "asyncio"
    
    def test_uvloop_python312_version_check(self):
        """Python 3.12 should require uvloop >= 0.19.0."""
        import sys
        
        # Mock Python 3.12
        with patch.object(sys, "version_info", (3, 12, 0)):
            # Mock uvloop with old version
            mock_uvloop = MagicMock()
            mock_uvloop.__version__ = "0.18.0"
            
            with patch.dict("sys.modules", {"uvloop": mock_uvloop}):
                with patch("platform.system", return_value="Linux"):
                    # Should detect incompatibility
                    assert not _is_uvloop_python312_compatible()
            
            # Mock uvloop with new version
            mock_uvloop.__version__ = "0.19.0"
            with patch.dict("sys.modules", {"uvloop": mock_uvloop}):
                with patch("platform.system", return_value="Linux"):
                    # Should be compatible
                    assert _is_uvloop_python312_compatible()
    
    def test_uvloop_auto_fallback_python312(self):
        """Auto loop should fall back to asyncio on Python 3.12 with old uvloop."""
        import sys
        
        mock_uvloop = MagicMock()
        mock_uvloop.__version__ = "0.18.0"
        
        with patch.object(sys, "version_info", (3, 12, 0)):
            with patch.dict("sys.modules", {"uvloop": mock_uvloop}):
                with patch("platform.system", return_value="Linux"):
                    with patch.object(RuntimeParameters, "has", return_value=False):
                        config = UvicornConfig.from_runtime_params()
        
        # Should fall back to asyncio due to uvloop version
        assert config.loop == "asyncio"
    
    def test_ssl_validation(self, tmp_path):
        """SSL files should be validated."""
        cert_file = tmp_path / "cert.pem"
        cert_file.touch()
        
        with patch.object(RuntimeParameters, "has", side_effect=lambda x: x == "DRUM_SSL_CERTFILE"):
            with patch.object(RuntimeParameters, "get", return_value=str(cert_file)):
                # Should not raise for existing file
                config = UvicornConfig.from_runtime_params()
        
        with patch.object(RuntimeParameters, "has", side_effect=lambda x: x == "DRUM_SSL_CERTFILE"):
            with patch.object(RuntimeParameters, "get", return_value="/nonexistent/cert.pem"):
                with pytest.raises(ValueError, match="not found"):
                    UvicornConfig.from_runtime_params()
```
