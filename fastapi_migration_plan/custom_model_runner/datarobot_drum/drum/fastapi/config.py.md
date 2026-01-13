# Plan: custom_model_runner/datarobot_drum/drum/fastapi/config.py

Configuration handler for Uvicorn parameters, mirroring `gunicorn.conf.py`.

## Overview

This module extracts runtime parameters and maps them to Uvicorn configuration. It provides a unified configuration interface for the Uvicorn server.

## Proposed Implementation:

```python
"""
Uvicorn configuration module.
Maps DRUM RuntimeParameters to Uvicorn settings.
"""
import os
from dataclasses import dataclass
from typing import Optional

from datarobot_drum import RuntimeParameters


@dataclass
class UvicornConfig:
    """Configuration for Uvicorn server."""
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    timeout_keep_alive: int = 5
    timeout_graceful_shutdown: Optional[int] = None
    backlog: int = 2048
    log_level: str = "info"
    loop: str = "auto"  # "auto", "asyncio", "uvloop"
    limit_max_requests: Optional[int] = None
    executor_workers: int = 4
    max_upload_size: int = 100 * 1024 * 1024  # Default 100MB
    
    # SSL/TLS Configuration
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    ssl_keyfile_password: Optional[str] = None
    ssl_version: int = 2  # ssl.PROTOCOL_TLS_SERVER
    ssl_cert_reqs: int = 0  # ssl.CERT_NONE
    ssl_ca_certs: Optional[str] = None
    ssl_ciphers: str = "TLSv1"
    
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
        
        # Timeout (equivalent to DRUM_CLIENT_REQUEST_TIMEOUT)
        # Note: Uvicorn uses timeout_keep_alive differently than gunicorn's timeout
        if RuntimeParameters.has("DRUM_CLIENT_REQUEST_TIMEOUT"):
            temp_timeout = int(RuntimeParameters.get("DRUM_CLIENT_REQUEST_TIMEOUT"))
            if 0 <= temp_timeout <= 3600:
                config.timeout_keep_alive = temp_timeout
        
        # Max requests (equivalent to DRUM_GUNICORN_MAX_REQUESTS)
        if RuntimeParameters.has("DRUM_UVICORN_MAX_REQUESTS"):
            temp_max_requests = int(RuntimeParameters.get("DRUM_UVICORN_MAX_REQUESTS"))
            if 0 <= temp_max_requests <= 10000:
                config.limit_max_requests = temp_max_requests
        elif RuntimeParameters.has("DRUM_GUNICORN_MAX_REQUESTS"):
            # Fallback to gunicorn param for compatibility
            temp_max_requests = int(RuntimeParameters.get("DRUM_GUNICORN_MAX_REQUESTS"))
            if 0 <= temp_max_requests <= 10000:
                config.limit_max_requests = temp_max_requests
        
        # Graceful shutdown timeout
        if RuntimeParameters.has("DRUM_UVICORN_GRACEFUL_TIMEOUT"):
            temp_graceful_timeout = int(RuntimeParameters.get("DRUM_UVICORN_GRACEFUL_TIMEOUT"))
            if 1 <= temp_graceful_timeout <= 3600:
                config.timeout_graceful_shutdown = temp_graceful_timeout
        elif RuntimeParameters.has("DRUM_GUNICORN_GRACEFUL_TIMEOUT"):
            # Fallback to gunicorn param for compatibility
            temp_graceful_timeout = int(RuntimeParameters.get("DRUM_GUNICORN_GRACEFUL_TIMEOUT"))
            if 1 <= temp_graceful_timeout <= 3600:
                config.timeout_graceful_shutdown = temp_graceful_timeout
        
        # Keep alive (equivalent to DRUM_GUNICORN_KEEP_ALIVE)
        if RuntimeParameters.has("DRUM_UVICORN_KEEP_ALIVE"):
            temp_keepalive = int(RuntimeParameters.get("DRUM_UVICORN_KEEP_ALIVE"))
            if 1 <= temp_keepalive <= 3600:
                config.timeout_keep_alive = temp_keepalive
        elif RuntimeParameters.has("DRUM_GUNICORN_KEEP_ALIVE"):
            temp_keepalive = int(RuntimeParameters.get("DRUM_GUNICORN_KEEP_ALIVE"))
            if 1 <= temp_keepalive <= 3600:
                config.timeout_keep_alive = temp_keepalive
        
        # Log level
        if RuntimeParameters.has("DRUM_UVICORN_LOG_LEVEL"):
            temp_loglevel = str(RuntimeParameters.get("DRUM_UVICORN_LOG_LEVEL")).lower()
            if temp_loglevel in {"debug", "info", "warning", "error", "critical", "trace"}:
                config.log_level = temp_loglevel
        elif RuntimeParameters.has("DRUM_GUNICORN_LOG_LEVEL"):
            temp_loglevel = str(RuntimeParameters.get("DRUM_GUNICORN_LOG_LEVEL")).lower()
            if temp_loglevel in {"debug", "info", "warning", "error", "critical"}:
                config.log_level = temp_loglevel
        
        # Event loop implementation
        if RuntimeParameters.has("DRUM_UVICORN_LOOP"):
            temp_loop = str(RuntimeParameters.get("DRUM_UVICORN_LOOP")).lower()
            if temp_loop in {"auto", "asyncio", "uvloop"}:
                config.loop = temp_loop
        
        # Executor workers for sync predictions
        if RuntimeParameters.has("DRUM_FASTAPI_EXECUTOR_WORKERS"):
            config.executor_workers = int(RuntimeParameters.get("DRUM_FASTAPI_EXECUTOR_WORKERS"))
        
        # Max upload size
        if RuntimeParameters.has("DRUM_FASTAPI_MAX_UPLOAD_SIZE"):
            config.max_upload_size = int(RuntimeParameters.get("DRUM_FASTAPI_MAX_UPLOAD_SIZE"))

        # SSL Configuration
        if RuntimeParameters.has("DRUM_SSL_CERTFILE"):
            config.ssl_certfile = str(RuntimeParameters.get("DRUM_SSL_CERTFILE"))
        if RuntimeParameters.has("DRUM_SSL_KEYFILE"):
            config.ssl_keyfile = str(RuntimeParameters.get("DRUM_SSL_KEYFILE"))
        if RuntimeParameters.has("DRUM_SSL_KEYFILE_PASSWORD"):
            config.ssl_keyfile_password = str(RuntimeParameters.get("DRUM_SSL_KEYFILE_PASSWORD"))
        if RuntimeParameters.has("DRUM_SSL_VERSION"):
            config.ssl_version = int(RuntimeParameters.get("DRUM_SSL_VERSION"))
        if RuntimeParameters.has("DRUM_SSL_CERT_REQS"):
            config.ssl_cert_reqs = int(RuntimeParameters.get("DRUM_SSL_CERT_REQS"))
        if RuntimeParameters.has("DRUM_SSL_CA_CERTS"):
            config.ssl_ca_certs = str(RuntimeParameters.get("DRUM_SSL_CA_CERTS"))
        if RuntimeParameters.has("DRUM_SSL_CIPHERS"):
            config.ssl_ciphers = str(RuntimeParameters.get("DRUM_SSL_CIPHERS"))
        
        return config
    
    def to_uvicorn_kwargs(self) -> dict:
        """Convert config to kwargs for uvicorn.run()."""
        kwargs = {
            "host": self.host,
            "port": self.port,
            "workers": self.workers,
            "timeout_keep_alive": self.timeout_keep_alive,
            "backlog": self.backlog,
            "log_level": self.log_level,
            "loop": self.loop,
            "access_log": True,
        }
        
        if self.timeout_graceful_shutdown is not None:
            kwargs["timeout_graceful_shutdown"] = self.timeout_graceful_shutdown
        
        if self.limit_max_requests is not None and self.limit_max_requests > 0:
            kwargs["limit_max_requests"] = self.limit_max_requests

        # SSL support in uvicorn.run()
        if self.ssl_certfile:
            kwargs["ssl_certfile"] = self.ssl_certfile
            kwargs["ssl_keyfile"] = self.ssl_keyfile
            kwargs["ssl_keyfile_password"] = self.ssl_keyfile_password
            kwargs["ssl_version"] = self.ssl_version
            kwargs["ssl_cert_reqs"] = self.ssl_cert_reqs
            kwargs["ssl_ca_certs"] = self.ssl_ca_certs
            kwargs["ssl_ciphers"] = self.ssl_ciphers
        
        return kwargs
    
    def to_cli_args(self) -> list:
        """Convert config to Uvicorn CLI arguments."""
        args = [
            "--host", self.host,
            "--port", str(self.port),
            "--workers", str(self.workers),
            "--timeout-keep-alive", str(self.timeout_keep_alive),
            "--backlog", str(self.backlog),
            "--log-level", self.log_level,
            "--loop", self.loop,
        ]
        
        if self.timeout_graceful_shutdown is not None:
            args.extend(["--timeout-graceful-shutdown", str(self.timeout_graceful_shutdown)])
        
        if self.limit_max_requests is not None and self.limit_max_requests > 0:
            args.extend(["--limit-max-requests", str(self.limit_max_requests)])

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
```

## Parameter Mapping Table

| DRUM Parameter | Gunicorn Equivalent | Uvicorn Equivalent | Default |
|----------------|---------------------|-------------------|---------|
| `CUSTOM_MODEL_WORKERS` | `workers` | `workers` | 1 |
| `DRUM_WEBSERVER_BACKLOG` | `backlog` | `backlog` | 2048 |
| `DRUM_CLIENT_REQUEST_TIMEOUT` | `timeout` | `timeout_keep_alive` | 120 |
| `DRUM_UVICORN_MAX_REQUESTS` | `max_requests` | `limit_max_requests` | None |
| `DRUM_UVICORN_GRACEFUL_TIMEOUT` | `graceful_timeout` | `timeout_graceful_shutdown` | None |
| `DRUM_UVICORN_KEEP_ALIVE` | `keepalive` | `timeout_keep_alive` | 5 |
| `DRUM_UVICORN_LOG_LEVEL` | `loglevel` | `log_level` | "info" |
| `DRUM_UVICORN_LOOP` | N/A (gevent) | `loop` | "auto" |
| `DRUM_FASTAPI_EXECUTOR_WORKERS` | N/A | `executor_workers` | 4 |
| `DRUM_FASTAPI_MAX_UPLOAD_SIZE` | N/A | `max_upload_size` | 104857600 |

## Notes:
- Uvicorn does not have a direct equivalent to gunicorn's `worker_class` (sync/gevent). Instead, it uses `loop` (asyncio/uvloop).
- The `max_requests_jitter` parameter from gunicorn is not directly supported by Uvicorn.
- Fallback to `DRUM_GUNICORN_*` parameters is provided for backward compatibility.
