# Plan: custom_model_runner/datarobot_drum/drum/fastapi/routes.py

Core route definitions for FastAPI, abstracted from logic, including Prometheus metrics endpoint.

## Proposed Implementation:

```python
"""
Core route definitions for FastAPI DRUM server.

Includes:
- Health check endpoints (/ping, /health, /livez, /readyz, /startupz)
- Chat completions with SSE streaming support
- Prometheus metrics endpoint (/metrics)
- Info endpoint (/info)
"""
import asyncio
import json
import time
import logging
from typing import Optional, Any, AsyncGenerator

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from datarobot_drum.drum.fastapi.app import get_worker_ctx
from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)

router = APIRouter()


# =============================================================================
# Pydantic Response Models (for OpenAPI documentation)
# =============================================================================

class PredictionResponse(BaseModel):
    """Response model for prediction endpoints."""
    predictions: list[Any] = Field(..., description="Model predictions")

class ErrorResponse(BaseModel):
    """Standard error response model."""
    message: str = Field(..., description="Error message")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    error: Optional[str] = Field(None, description="Error code")

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Health status")
    message: Optional[str] = Field(None, description="Additional message")
    reason: Optional[str] = Field(None, description="Reason if not healthy")


# =============================================================================
# Health Check Endpoints
# =============================================================================

@router.get("/")
@router.get("/ping/")
@router.get("/ping")
async def ping():
    """Liveness probe - returns OK if server is running."""
    ctx = get_worker_ctx()
    if ctx and hasattr(ctx, "predictor") and hasattr(ctx.predictor, "liveness_probe"):
        return ctx.predictor.liveness_probe()
    return {"message": "OK"}


@router.get("/health/")
@router.get("/health")
async def health():
    """Legacy readiness probe - returns OK if model is loaded."""
    ctx = get_worker_ctx()
    if ctx and hasattr(ctx, "predictor") and hasattr(ctx.predictor, "readiness_probe"):
        return ctx.predictor.readiness_probe()
    return {"message": "OK"}


@router.get("/livez")
async def livez():
    """
    Kubernetes liveness probe.
    
    Returns 200 if the server process is alive.
    Does NOT check model health (that's readyz).
    """
    return {"status": "alive"}


@router.get("/readyz")
async def readyz():
    """
    Kubernetes readiness probe.
    
    Returns 200 if:
    - Worker context is running
    - Model is loaded and ready to accept requests
    
    Returns 503 if not ready.
    """
    ctx = get_worker_ctx()
    
    if not ctx or not ctx.is_running:
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "reason": "worker not running"}
        )
    
    if hasattr(ctx, "predictor") and hasattr(ctx.predictor, "readiness_probe"):
        result = ctx.predictor.readiness_probe()
        if isinstance(result, dict) and result.get("status") != "ok":
            return JSONResponse(status_code=503, content=result)
    
    return {"status": "ready"}


@router.get("/startupz")
async def startupz():
    """
    Kubernetes startup probe.
    
    Returns 200 only after model is fully loaded.
    Use with startupProbe in K8s to allow slow model loading
    without affecting liveness/readiness probe intervals.
    
    Example K8s config:
        startupProbe:
          httpGet:
            path: /startupz
            port: 8080
          failureThreshold: 30
          periodSeconds: 10
    """
    ctx = get_worker_ctx()
    
    if not ctx:
        return JSONResponse(
            status_code=503, 
            content={"status": "initializing", "model_loaded": False}
        )
    
    if not getattr(ctx, "model_loaded", False):
        return JSONResponse(
            status_code=503, 
            content={"status": "loading model", "model_loaded": False}
        )
    
    return {"status": "started", "model_loaded": True}


# =============================================================================
# Chat Completions with SSE Streaming
# =============================================================================

async def _stream_chat_response(
    predictor, 
    body: dict, 
    request: Request
) -> AsyncGenerator[str, None]:
    """
    Generator for SSE streaming chat responses.
    
    Yields Server-Sent Events in OpenAI-compatible format:
    - data: {chunk JSON}
    - data: [DONE] at end
    
    Handles client disconnection gracefully.
    """
    try:
        # Check if predictor supports streaming
        if not hasattr(predictor, "stream_chat"):
            error_chunk = {
                "error": {
                    "message": "Streaming not supported by this model",
                    "type": "unsupported_operation"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            return
        
        async for chunk in predictor.stream_chat(body):
            # Check if client disconnected
            if await request.is_disconnected():
                logger.info("Stream cancelled - client disconnected")
                return
            
            yield f"data: {json.dumps(chunk)}\n\n"
        
        yield "data: [DONE]\n\n"
        
    except asyncio.CancelledError:
        logger.info("Stream cancelled by client")
        raise
    except Exception as e:
        logger.error("Error during streaming: %s", e)
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "internal_error"
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"


@router.post("/chat/completions")
@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible chat completions endpoint.
    
    Supports both streaming (SSE) and non-streaming responses.
    Set "stream": true in request body for streaming.
    
    Request body:
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false,  // Optional, default false
            "model": "...",   // Optional
            ...
        }
    
    Streaming response:
        Content-Type: text/event-stream
        data: {"choices": [{"delta": {"content": "..."}}]}
        data: [DONE]
    
    Non-streaming response:
        Content-Type: application/json
        {"choices": [{"message": {"content": "..."}}]}
    """
    ctx = get_worker_ctx()
    
    if not ctx:
        return JSONResponse(
            status_code=503,
            content={"error": {"message": "Model not ready", "type": "service_unavailable"}}
        )
    
    try:
        body = await request.json()
    except json.JSONDecodeError as e:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Invalid JSON: {e}", "type": "invalid_request"}}
        )
    
    # Check if streaming is requested
    stream = body.get("stream", False)
    
    if stream:
        return StreamingResponse(
            _stream_chat_response(ctx.predictor, body, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            }
        )
    
    # Non-streaming response
    if hasattr(ctx.predictor, "chat_completion"):
        try:
            result = await ctx.predictor.chat_completion(body)
            return JSONResponse(content=result)
        except Exception as e:
            logger.error("Chat completion error: %s", e)
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(e), "type": "internal_error"}}
            )
    else:
        return JSONResponse(
            status_code=501,
            content={"error": {"message": "Chat completions not supported", "type": "not_implemented"}}
        )


# =============================================================================
# Prediction Endpoints with OpenAPI Schema
# =============================================================================

@router.post(
    "/predict/",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input data"},
        503: {"model": ErrorResponse, "description": "Model not ready"},
        504: {"model": ErrorResponse, "description": "Request timeout"},
    },
    summary="Make predictions",
    description="Submit data for model predictions. Accepts CSV or JSON input.",
    tags=["predictions"],
)
@router.post("/predict")
async def predict(request: Request):
    """
    Main prediction endpoint.
    
    Accepts CSV or JSON input data and returns model predictions.
    """
    ctx = get_worker_ctx()
    
    if not ctx:
        return JSONResponse(
            status_code=503,
            content={"message": "Model not ready", "error": "service_unavailable"}
        )
    
    if hasattr(ctx, "predictor") and hasattr(ctx.predictor, "predict"):
        try:
            result = await ctx.predictor.predict(request)
            return result
        except Exception as e:
            logger.error("Prediction error: %s", e)
            return JSONResponse(
                status_code=500,
                content={"message": str(e), "error": "prediction_error"}
            )
    
    return JSONResponse(
        status_code=501,
        content={"message": "Prediction not supported", "error": "not_implemented"}
    )


# =============================================================================
# Prometheus Metrics Endpoint
# =============================================================================

# Initialize Prometheus metrics (lazy import to avoid dependency issues)
_prometheus_initialized = False
_metrics = {}


def _init_prometheus_metrics():
    """Initialize Prometheus metrics collectors."""
    global _prometheus_initialized, _metrics
    
    if _prometheus_initialized:
        return
    
    try:
        from prometheus_client import Counter, Histogram, Gauge, Info
        
        _metrics["requests_total"] = Counter(
            "drum_requests_total",
            "Total number of requests",
            ["method", "endpoint", "status"]
        )
        
        _metrics["request_duration_seconds"] = Histogram(
            "drum_request_duration_seconds",
            "Request duration in seconds",
            ["method", "endpoint"],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
        )
        
        _metrics["predictions_in_progress"] = Gauge(
            "drum_predictions_in_progress",
            "Number of predictions currently being processed"
        )
        
        _metrics["memory_bytes"] = Gauge(
            "drum_memory_bytes",
            "Current memory usage in bytes",
            ["type"]  # rss, vms, shared
        )
        
        _metrics["executor_queue_depth"] = Gauge(
            "drum_executor_queue_depth",
            "Number of requests waiting in executor queue"
        )
        
        _metrics["executor_workers_busy"] = Gauge(
            "drum_executor_workers_busy",
            "Number of executor workers currently processing requests"
        )
        
        _metrics["model_info"] = Info(
            "drum_model",
            "Information about the loaded model"
        )
        
        _prometheus_initialized = True
        logger.info("Prometheus metrics initialized")
        
    except ImportError:
        logger.warning(
            "prometheus_client not installed. /metrics endpoint will return 501. "
            "Install with: pip install prometheus-client"
        )


def _update_memory_metrics():
    """Update memory metrics from psutil."""
    if "memory_bytes" not in _metrics:
        return
    
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        
        _metrics["memory_bytes"].labels(type="rss").set(mem_info.rss)
        _metrics["memory_bytes"].labels(type="vms").set(mem_info.vms)
        if hasattr(mem_info, "shared"):
            _metrics["memory_bytes"].labels(type="shared").set(mem_info.shared)
    except Exception as e:
        logger.debug("Failed to update memory metrics: %s", e)


def _update_executor_metrics():
    """Update executor metrics from worker context."""
    ctx = get_worker_ctx()
    if not ctx:
        return
    
    # Get executor metrics if available
    if hasattr(ctx, "prediction_server") and hasattr(ctx.prediction_server, "_executor"):
        executor = ctx.prediction_server._executor
        if hasattr(executor, "_pending_count"):
            if "executor_queue_depth" in _metrics:
                _metrics["executor_queue_depth"].set(executor._pending_count)


@router.get("/metrics")
@router.get("/metrics/")
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus text format for scraping.
    
    Metrics exposed:
    - drum_requests_total: Counter of total requests by method, endpoint, status
    - drum_request_duration_seconds: Histogram of request durations
    - drum_predictions_in_progress: Gauge of in-flight predictions
    - drum_memory_bytes: Gauge of memory usage (rss, vms, shared)
    - drum_executor_queue_depth: Gauge of executor queue depth
    - drum_model_info: Info about the loaded model
    """
    _init_prometheus_metrics()
    
    if not _prometheus_initialized:
        return JSONResponse(
            status_code=501,
            content={
                "message": "Prometheus metrics not available",
                "reason": "prometheus_client not installed"
            }
        )
    
    # Update point-in-time metrics
    _update_memory_metrics()
    _update_executor_metrics()
    
    # Generate Prometheus format output
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# =============================================================================
# Prometheus Middleware (for request counting)
# =============================================================================

class PrometheusMiddleware:
    """
    Middleware to record request metrics for Prometheus.
    
    Records:
    - Request count by method, endpoint, status
    - Request duration histogram
    - In-flight predictions gauge
    """
    
    def __init__(self, app):
        self.app = app
        _init_prometheus_metrics()
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        
        # Skip metrics for health endpoints to reduce noise
        path = scope.get("path", "")
        if path.rstrip("/") in ["/ping", "/health", "/livez", "/readyz", "/metrics"]:
            return await self.app(scope, receive, send)
        
        method = scope.get("method", "GET")
        endpoint = self._normalize_endpoint(path)
        
        # Track in-flight requests
        if _prometheus_initialized and "predictions_in_progress" in _metrics:
            _metrics["predictions_in_progress"].inc()
        
        start_time = time.perf_counter()
        status_code = 500  # Default in case of error
        
        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 500)
            await send(message)
        
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.perf_counter() - start_time
            
            if _prometheus_initialized:
                if "requests_total" in _metrics:
                    _metrics["requests_total"].labels(
                        method=method,
                        endpoint=endpoint,
                        status=str(status_code)
                    ).inc()
                
                if "request_duration_seconds" in _metrics:
                    _metrics["request_duration_seconds"].labels(
                        method=method,
                        endpoint=endpoint
                    ).observe(duration)
                
                if "predictions_in_progress" in _metrics:
                    _metrics["predictions_in_progress"].dec()
    
    @staticmethod
    def _normalize_endpoint(path: str) -> str:
        """
        Normalize endpoint path for metric labels.
        
        Prevents high cardinality by grouping similar paths.
        """
        path = path.rstrip("/")
        
        # Group prediction endpoints
        if path.startswith("/predict"):
            return "/predict"
        if path.startswith("/transform"):
            return "/transform"
        if path.startswith("/invocations"):
            return "/invocations"
        if path.startswith("/predictionsUnstructured") or path.startswith("/predictUnstructured"):
            return "/predictUnstructured"
        
        return path or "/"
```

## Usage

Add the Prometheus middleware in `app.py`:

```python
# In app.py - create_app()
from datarobot_drum.drum.fastapi.routes import router, PrometheusMiddleware

def create_app() -> FastAPI:
    app = FastAPI(...)
    
    # Add Prometheus middleware
    app.add_middleware(PrometheusMiddleware)
    
    # Include routes
    app.include_router(router)
    
    return app
```

## Grafana Dashboard Queries

Example PromQL queries for monitoring:

```yaml
# Request rate
sum(rate(drum_requests_total[5m])) by (endpoint)

# Error rate
sum(rate(drum_requests_total{status=~"5.."}[5m])) / sum(rate(drum_requests_total[5m]))

# P99 latency
histogram_quantile(0.99, sum(rate(drum_request_duration_seconds_bucket[5m])) by (le, endpoint))

# Memory usage
drum_memory_bytes{type="rss"}

# Queue depth (backpressure indicator)
drum_executor_queue_depth
```
