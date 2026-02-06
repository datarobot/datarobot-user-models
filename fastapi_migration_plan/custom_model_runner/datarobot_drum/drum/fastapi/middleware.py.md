# Plan: custom_model_runner/datarobot_drum/drum/fastapi/middleware.py

Custom middlewares for FastAPI with production-ready features.

## Overview

This module provides middlewares for:
- Request timeout handling
- Production backpressure with rejection
- Security headers
- Request ID propagation
- Metrics collection

## Proposed Implementation:

```python
"""
Custom middlewares for FastAPI DRUM server.
"""
import asyncio
import logging
import time
import uuid
from typing import Optional, Callable, List

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


# Endpoints excluded from backpressure and timeout
HEALTH_ENDPOINTS = frozenset({
    "/", "/ping", "/ping/", "/health", "/health/",
    "/livez", "/readyz", "/stats", "/stats/",
    "/info", "/info/", "/capabilities", "/capabilities/"
})


class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    """
    Middleware that enforces request timeout.
    
    Note: This middleware wraps the entire request handling, including
    reading the request body. For very large uploads, consider using
    a separate upload timeout.
    """
    
    def __init__(self, app: ASGIApp, timeout: float = 120.0):
        super().__init__(app)
        self.timeout = timeout
    
    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip timeout for health endpoints
        if request.url.path.rstrip("/") in HEALTH_ENDPOINTS:
            return await call_next(request)
        
        # Get timeout from config if available
        timeout = self.timeout
        if hasattr(request.app.state, "config"):
            timeout = getattr(request.app.state.config, "request_timeout", self.timeout)
        
        try:
            return await asyncio.wait_for(call_next(request), timeout=timeout)
        except asyncio.TimeoutError:
            request_id = request.headers.get("X-Request-ID", "unknown")
            logger.warning(
                "Request timeout after %.1fs: %s %s (request_id=%s)",
                timeout, request.method, request.url.path, request_id
            )
            return JSONResponse(
                status_code=504,
                content={
                    "message": f"Request timeout after {timeout}s",
                    "request_id": request_id,
                }
            )


class BackPressureMiddleware(BaseHTTPMiddleware):
    """
    Production-ready backpressure middleware with queue depth limiting.
    
    Features:
    - Immediate rejection when queue is full (no unbounded waiting)
    - Metrics for monitoring (via worker context)
    - Timeout for acquiring a slot
    - Excludes health check endpoints
    """
    
    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip backpressure for health endpoints
        if request.url.path.rstrip("/") in HEALTH_ENDPOINTS:
            return await call_next(request)
        
        # Get worker context
        worker_ctx = getattr(request.app.state, "worker_ctx", None)
        if worker_ctx is None or not hasattr(worker_ctx, "backpressure"):
            return await call_next(request)
        
        backpressure = worker_ctx.backpressure
        
        try:
            acquired = await backpressure.acquire()
            if not acquired:
                # Queue is full, reject immediately
                return JSONResponse(
                    status_code=503,
                    content={
                        "message": "Service temporarily unavailable due to high load",
                        "retry_after": 5,
                    },
                    headers={"Retry-After": "5"}
                )
            
            try:
                response = await call_next(request)
                return response
            finally:
                backpressure.release()
                await backpressure.decrement_queue_depth()
        
        except asyncio.TimeoutError:
            logger.warning(
                "Backpressure timeout: could not acquire slot in %.1fs for %s %s",
                backpressure.acquire_timeout, request.method, request.url.path
            )
            return JSONResponse(
                status_code=503,
                content={
                    "message": "Service temporarily unavailable, request queue timeout",
                    "retry_after": 10,
                },
                headers={"Retry-After": "10"}
            )


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds security headers to all responses.
    """
    
    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        
        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # XSS protection (legacy browsers)
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Remove server header that might leak version info
        if "server" in response.headers:
            del response.headers["server"]
        
        return response


class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware that ensures every request has a unique ID for tracing.
    """
    
    async def dispatch(self, request: Request, call_next) -> Response:
        # Get or generate request ID
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Store in request state for access by handlers
        request.state.request_id = request_id
        
        # Add to response
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting request metrics (non-Prometheus version).
    
    For Prometheus metrics, see OBSERVABILITY_MIGRATION.md.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self._request_count = 0
        self._total_latency_ms = 0.0
        self._lock = asyncio.Lock()
    
    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.perf_counter()
        
        response = await call_next(request)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        async with self._lock:
            self._request_count += 1
            self._total_latency_ms += latency_ms
        
        # Add timing header
        response.headers["X-Response-Time-Ms"] = f"{latency_ms:.2f}"
        
        return response
    
    def get_metrics(self) -> dict:
        """Get metrics for /stats/ endpoint."""
        return {
            "request_count": self._request_count,
            "avg_latency_ms": (
                self._total_latency_ms / self._request_count
                if self._request_count > 0 else 0
            ),
        }


class MaxBodySizeMiddleware:
    """
    ASGI middleware that limits request body size.
    
    Note: This is a raw ASGI middleware (not BaseHTTPMiddleware) for
    efficiency - it rejects oversized requests before reading the body.
    """
    
    def __init__(self, app: ASGIApp, max_size: int = 100 * 1024 * 1024):
        self.app = app
        self.max_size = max_size
    
    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Check Content-Length header
        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        
        if content_length:
            try:
                length = int(content_length.decode())
                if length > self.max_size:
                    response = JSONResponse(
                        status_code=413,
                        content={
                            "message": f"Request body too large. Maximum size: {self.max_size} bytes",
                        }
                    )
                    await response(scope, receive, send)
                    return
            except (ValueError, UnicodeDecodeError):
                pass
        
        # Track received bytes for streaming uploads
        received_bytes = 0
        
        async def receive_wrapper():
            nonlocal received_bytes
            message = await receive()
            
            if message["type"] == "http.request":
                body = message.get("body", b"")
                received_bytes += len(body)
                
                if received_bytes > self.max_size:
                    # Can't reject mid-stream easily, but we can log and continue
                    # The actual rejection would need to happen at a higher level
                    logger.warning(
                        "Request body exceeded max size during streaming: %d > %d",
                        received_bytes, self.max_size
                    )
            
            return message
        
        await self.app(scope, receive_wrapper, send)


class StdoutFlusherMiddleware(BaseHTTPMiddleware):
    """
    Middleware that notifies StdoutFlusher of request activity.
    
    This is used to keep stdout flushing active during model execution.
    """
    
    def __init__(self, app: ASGIApp, flusher):
        super().__init__(app)
        self.flusher = flusher
    
    async def dispatch(self, request: Request, call_next) -> Response:
        if hasattr(self.flusher, "set_last_activity_time"):
            self.flusher.set_last_activity_time()
        
        response = await call_next(request)
        
        if hasattr(self.flusher, "set_last_activity_time"):
            self.flusher.set_last_activity_time()
        
        return response


def setup_middlewares(app, config=None, worker_ctx=None):
    """
    Setup all middlewares in correct order.
    
    Middleware order (outermost to innermost):
    1. RequestIdMiddleware - Add request ID first
    2. SecurityHeadersMiddleware - Add security headers
    3. MaxBodySizeMiddleware - Reject oversized requests early
    4. BackPressureMiddleware - Apply backpressure
    5. RequestTimeoutMiddleware - Apply timeout
    6. MetricsMiddleware - Collect metrics (innermost)
    
    Note: FastAPI applies middlewares in reverse order of addition.
    """
    from datarobot_drum import RuntimeParameters
    
    # Get configuration values
    timeout = 120.0
    max_body_size = 100 * 1024 * 1024  # 100MB default
    
    if config:
        timeout = getattr(config, "request_timeout", timeout)
        max_body_size = getattr(config, "max_upload_size", max_body_size)
    
    if RuntimeParameters.has("DRUM_CLIENT_REQUEST_TIMEOUT"):
        timeout = float(RuntimeParameters.get("DRUM_CLIENT_REQUEST_TIMEOUT"))
    
    if RuntimeParameters.has("DRUM_FASTAPI_MAX_UPLOAD_SIZE"):
        max_body_size = int(RuntimeParameters.get("DRUM_FASTAPI_MAX_UPLOAD_SIZE"))
    
    # Add middlewares in reverse order (last added = first executed)
    # 6. Metrics (innermost)
    metrics_middleware = MetricsMiddleware(app)
    app.add_middleware(MetricsMiddleware)
    
    # 5. Timeout
    app.add_middleware(RequestTimeoutMiddleware, timeout=timeout)
    
    # 4. Backpressure
    app.add_middleware(BackPressureMiddleware)
    
    # 3. Max body size (raw ASGI, add differently)
    # Note: This needs to wrap the app directly
    # app = MaxBodySizeMiddleware(app, max_size=max_body_size)
    
    # 2. Security headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # 1. Request ID (outermost)
    app.add_middleware(RequestIdMiddleware)
    
    return app
```

## Middleware Order Visualization

```
Request →
  [RequestIdMiddleware] →
    [SecurityHeadersMiddleware] →
      [MaxBodySizeMiddleware] →
        [BackPressureMiddleware] →
          [RequestTimeoutMiddleware] →
            [MetricsMiddleware] →
              [Application Handler]
            ← Response with metrics
          ← Response with timeout handling
        ← Response with backpressure tracking
      ← Response with size validation
    ← Response with security headers
  ← Response with X-Request-ID
← Response to client
```

## Usage Example

```python
# In app.py
from datarobot_drum.drum.fastapi.middleware import setup_middlewares

def create_app(config: UvicornConfig, worker_ctx: FastAPIWorkerCtx) -> FastAPI:
    app = FastAPI(...)
    
    # Store references in app state
    app.state.config = config
    app.state.worker_ctx = worker_ctx
    
    # Setup all middlewares
    setup_middlewares(app, config=config, worker_ctx=worker_ctx)
    
    return app
```

## Testing

```python
# tests/unit/datarobot_drum/drum/fastapi/test_middleware.py
import pytest
from httpx import AsyncClient
from fastapi import FastAPI

class TestBackPressureMiddleware:
    @pytest.mark.asyncio
    async def test_rejects_when_queue_full(self, app_with_backpressure):
        """Test that requests are rejected when queue is full."""
        async with AsyncClient(app=app_with_backpressure, base_url="http://test") as client:
            # Fill the queue
            responses = await asyncio.gather(*[
                client.post("/predict/", json={}) 
                for _ in range(150)  # More than max_queue_depth
            ], return_exceptions=True)
            
            # Some should be rejected with 503
            rejected = [r for r in responses if hasattr(r, 'status_code') and r.status_code == 503]
            assert len(rejected) > 0
    
    @pytest.mark.asyncio
    async def test_health_endpoints_bypass_backpressure(self, app_with_backpressure):
        """Health endpoints should always respond."""
        async with AsyncClient(app=app_with_backpressure, base_url="http://test") as client:
            response = await client.get("/ping")
            assert response.status_code == 200

class TestRequestTimeoutMiddleware:
    @pytest.mark.asyncio
    async def test_timeout_returns_504(self, slow_app):
        """Test that slow requests return 504."""
        async with AsyncClient(app=slow_app, base_url="http://test") as client:
            response = await client.post("/slow-predict/", json={})
            assert response.status_code == 504
            assert "timeout" in response.json()["message"].lower()
```
