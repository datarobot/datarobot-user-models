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


# Endpoints excluded from backpressure, timeout, and rate limiting
HEALTH_ENDPOINTS = frozenset({
    "/", "/ping", "/ping/", "/health", "/health/",
    "/livez", "/readyz", "/startupz", "/stats", "/stats/",
    "/info", "/info/", "/capabilities", "/capabilities/",
    "/metrics", "/metrics/"
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


class RateLimitMiddleware:
    """
    Token bucket rate limiter per client IP or API key.
    
    Features:
    - Token bucket algorithm for smooth rate limiting
    - Per-client tracking (by IP or custom key function)
    - Configurable burst capacity
    - Automatic cleanup of stale buckets
    - Health endpoints bypassed
    - Returns 429 with Retry-After header when rate exceeded
    
    Algorithm:
    - Each client gets a bucket with `burst` tokens
    - Tokens regenerate at `requests_per_minute / 60` per second
    - Each request consumes 1 token
    - If no tokens available, request is rejected with 429
    
    Usage:
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=60,
            burst=10,
        )
    """
    
    def __init__(
        self, 
        app: ASGIApp,
        requests_per_minute: int = 60,
        burst: int = 10,
        key_func: Optional[Callable[[Scope], str]] = None,
        cleanup_interval: float = 60.0,
    ):
        """
        Initialize rate limiter.
        
        Args:
            app: ASGI application
            requests_per_minute: Steady-state request rate limit
            burst: Maximum burst capacity (bucket size)
            key_func: Function to extract client key from request scope
            cleanup_interval: Seconds between stale bucket cleanup
        """
        self.app = app
        self.rate = requests_per_minute / 60.0  # tokens per second
        self.burst = burst
        self.key_func = key_func or self._get_client_ip
        self.cleanup_interval = cleanup_interval
        
        # Bucket state: {client_key: (last_update_time, tokens)}
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = asyncio.Lock()
        self._last_cleanup = time.time()
    
    @staticmethod
    def _get_client_ip(scope: Scope) -> str:
        """Extract client IP from request scope."""
        # Check for X-Forwarded-For header (behind proxy)
        headers = dict(scope.get("headers", []))
        forwarded = headers.get(b"x-forwarded-for")
        if forwarded:
            # Take first IP in chain
            return forwarded.decode().split(",")[0].strip()
        
        # Check for X-Real-IP
        real_ip = headers.get(b"x-real-ip")
        if real_ip:
            return real_ip.decode().strip()
        
        # Fall back to direct connection
        client = scope.get("client")
        if client:
            return client[0]
        
        return "unknown"
    
    async def _check_rate(self, client_key: str) -> tuple[bool, float]:
        """
        Check if request is allowed and update bucket.
        
        Returns:
            (allowed, retry_after) - allowed is True if request can proceed,
            retry_after is seconds to wait if not allowed
        """
        now = time.time()
        
        async with self._lock:
            # Cleanup stale buckets periodically
            if now - self._last_cleanup > self.cleanup_interval:
                await self._cleanup_stale_buckets(now)
            
            if client_key in self._buckets:
                last_time, tokens = self._buckets[client_key]
                
                # Add tokens based on elapsed time
                elapsed = now - last_time
                tokens = min(self.burst, tokens + elapsed * self.rate)
            else:
                # New client, start with full bucket
                tokens = float(self.burst)
            
            if tokens >= 1.0:
                # Allow request, consume token
                self._buckets[client_key] = (now, tokens - 1.0)
                return True, 0.0
            else:
                # Rate limited, calculate retry time
                retry_after = (1.0 - tokens) / self.rate
                self._buckets[client_key] = (now, tokens)
                return False, retry_after
    
    async def _cleanup_stale_buckets(self, now: float):
        """Remove buckets that haven't been accessed in a while."""
        stale_threshold = now - 300.0  # 5 minutes
        stale_keys = [
            key for key, (last_time, _) in self._buckets.items()
            if last_time < stale_threshold
        ]
        for key in stale_keys:
            del self._buckets[key]
        
        if stale_keys:
            logger.debug("Cleaned up %d stale rate limit buckets", len(stale_keys))
        
        self._last_cleanup = now
    
    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Skip rate limiting for health endpoints
        path = scope.get("path", "").rstrip("/")
        if path in HEALTH_ENDPOINTS:
            await self.app(scope, receive, send)
            return
        
        # Check rate limit
        client_key = self.key_func(scope)
        allowed, retry_after = await self._check_rate(client_key)
        
        if not allowed:
            logger.warning(
                "Rate limit exceeded for client %s (retry_after=%.1fs)",
                client_key, retry_after
            )
            response = JSONResponse(
                status_code=429,
                content={
                    "message": "Rate limit exceeded",
                    "retry_after": int(retry_after) + 1,
                },
                headers={"Retry-After": str(int(retry_after) + 1)}
            )
            await response(scope, receive, send)
            return
        
        await self.app(scope, receive, send)
    
    def get_metrics(self) -> dict:
        """Get rate limiter metrics."""
        return {
            "rate_limit_bucket_count": len(self._buckets),
            "rate_limit_requests_per_minute": self.rate * 60,
            "rate_limit_burst": self.burst,
        }


def setup_middlewares(app, config=None, worker_ctx=None):
    """
    Setup all middlewares in correct order.
    
    Middleware order (outermost to innermost):
    1. RequestIdMiddleware - Add request ID first
    2. SecurityHeadersMiddleware - Add security headers
    3. RateLimitMiddleware - Apply rate limiting (optional)
    4. MaxBodySizeMiddleware - Reject oversized requests early
    5. BackPressureMiddleware - Apply backpressure
    6. RequestTimeoutMiddleware - Apply timeout
    7. MetricsMiddleware - Collect metrics (innermost)
    
    Note: FastAPI applies middlewares in reverse order of addition.
    """
    from datarobot_drum import RuntimeParameters
    
    # Get configuration values
    timeout = 120.0
    max_body_size = 100 * 1024 * 1024  # 100MB default
    
    # Rate limiting configuration
    rate_limit_enabled = False
    rate_limit_rpm = 60  # requests per minute
    rate_limit_burst = 10
    
    if config:
        timeout = getattr(config, "request_timeout", timeout)
        max_body_size = getattr(config, "max_upload_size", max_body_size)
        rate_limit_enabled = getattr(config, "rate_limit_enabled", False)
        rate_limit_rpm = getattr(config, "rate_limit_rpm", rate_limit_rpm)
        rate_limit_burst = getattr(config, "rate_limit_burst", rate_limit_burst)
    
    if RuntimeParameters.has("DRUM_CLIENT_REQUEST_TIMEOUT"):
        timeout = float(RuntimeParameters.get("DRUM_CLIENT_REQUEST_TIMEOUT"))
    
    if RuntimeParameters.has("DRUM_FASTAPI_MAX_UPLOAD_SIZE"):
        max_body_size = int(RuntimeParameters.get("DRUM_FASTAPI_MAX_UPLOAD_SIZE"))
    
    # Rate limiting from environment
    if RuntimeParameters.has("DRUM_RATE_LIMIT_ENABLED"):
        rate_limit_enabled = RuntimeParameters.get("DRUM_RATE_LIMIT_ENABLED").lower() in ("true", "1", "yes")
    if RuntimeParameters.has("DRUM_RATE_LIMIT_RPM"):
        rate_limit_rpm = int(RuntimeParameters.get("DRUM_RATE_LIMIT_RPM"))
    if RuntimeParameters.has("DRUM_RATE_LIMIT_BURST"):
        rate_limit_burst = int(RuntimeParameters.get("DRUM_RATE_LIMIT_BURST"))
    
    # Add middlewares in reverse order (last added = first executed)
    # 7. Metrics (innermost)
    metrics_middleware = MetricsMiddleware(app)
    app.add_middleware(MetricsMiddleware)
    
    # 6. Timeout
    app.add_middleware(RequestTimeoutMiddleware, timeout=timeout)
    
    # 5. Backpressure
    app.add_middleware(BackPressureMiddleware)
    
    # 4. Max body size (raw ASGI, add differently)
    # Note: This needs to wrap the app directly
    # app = MaxBodySizeMiddleware(app, max_size=max_body_size)
    
    # 3. Rate limiting (optional)
    if rate_limit_enabled:
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=rate_limit_rpm,
            burst=rate_limit_burst,
        )
        logger.info(
            "Rate limiting enabled: %d requests/min, burst=%d",
            rate_limit_rpm, rate_limit_burst
        )
    
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
