# Observability Migration: Flask/Gunicorn → FastAPI/Uvicorn

This document ensures observability parity between the old and new server implementations.

---

## Log Format Changes

### Access Log Format

#### Gunicorn (Current)
```
%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"
```
**Example:**
```
127.0.0.1 - - [13/Jan/2026:10:15:32 +0000] "POST /predict/ HTTP/1.1" 200 1234 "-" "python-requests/2.28.0"
```

#### Uvicorn (New)
```
%(client_addr)s - "%(request_line)s" %(status_code)s
```
**Example:**
```
127.0.0.1:54321 - "POST /predict/ HTTP/1.1" 200
```

### Unified Access Log Format

To maintain log parsing compatibility, configure Uvicorn with a custom access log format:

```python
# In config.py
UVICORN_ACCESS_LOG_FORMAT = (
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(B)s "%(f)s" "%(a)s" %(D)s'
)

# In run_uvicorn.py
import uvicorn

class DrumAccessLogFormatter(uvicorn.logging.AccessFormatter):
    """Custom access log formatter matching Gunicorn format."""
    
    def formatMessage(self, record):
        # Map uvicorn record attributes to gunicorn-style
        record.h = record.client_addr.split(":")[0] if record.client_addr else "-"
        record.l = "-"
        record.u = "-"
        record.t = self.formatTime(record, "[%d/%b/%Y:%H:%M:%S %z]")
        record.r = f"{record.method} {record.path} HTTP/{record.http_version}"
        record.s = record.status_code
        record.B = record.bytes if hasattr(record, 'bytes') else "-"
        record.f = record.headers.get("referer", "-") if hasattr(record, 'headers') else "-"
        record.a = record.headers.get("user-agent", "-") if hasattr(record, 'headers') else "-"
        record.D = int(record.duration * 1_000_000) if hasattr(record, 'duration') else 0
        
        return UVICORN_ACCESS_LOG_FORMAT % record.__dict__
```

### Application Log Format

Both servers use the same DRUM logger configuration:

```python
# Shared log format (no changes needed)
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
```

**Key Logger Names:**
| Logger | Purpose |
|--------|---------|
| `drum.server` | Server lifecycle events |
| `drum.predictor` | Model loading and prediction |
| `drum.monitor` | Resource monitoring |
| `uvicorn.access` | HTTP access logs (new) |
| `uvicorn.error` | Server errors (new) |

---

## Metrics Mapping

### /stats/ Endpoint Response

Ensure the `/stats/` endpoint returns identical keys:

| Metric Key | Flask Value Source | FastAPI Value Source | Notes |
|------------|-------------------|---------------------|-------|
| `mem_info` | `psutil.Process().memory_info()` | Same | |
| `time_info` | `time.time()` | Same | |
| `container_info` | cgroups detection | Same (v1 + v2) | |
| `prediction_count` | Thread-safe counter | Same | |
| `request_duration_ms` | Manual timing | Middleware timing | |
| `model_info` | Predictor metadata | Same | |

**Parity Test:**
```python
def test_stats_parity():
    flask_stats = get_flask_server_stats()
    fastapi_stats = get_fastapi_server_stats()
    
    required_keys = {
        "mem_info", "time_info", "container_info", 
        "prediction_count", "model_info"
    }
    
    assert required_keys <= set(flask_stats.keys())
    assert required_keys <= set(fastapi_stats.keys())
```

---

## OpenTelemetry (OTel) Parity

### Span Names and Attributes

| Span Name | Flask Implementation | FastAPI Implementation |
|-----------|---------------------|------------------------|
| `drum.invocations` | Manual `tracer.start_as_current_span()` | Same |
| `drum.transform` | Manual | Same |
| `drum.predictUnstructured` | Manual | Same |
| HTTP span | Flask-OTEL auto-instrumentation | FastAPI-OTEL auto-instrumentation |

### Required Span Attributes

```python
# Both implementations must set these attributes
span.set_attribute("drum.model_id", model_id)
span.set_attribute("drum.target_type", target_type)
span.set_attribute("drum.language", language)
span.set_attribute("http.request.body.size", content_length)
span.set_attribute("http.response.body.size", response_length)
```

### OTel Instrumentation Setup

```python
# In otel.py
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

def setup_otel_instrumentation(app: FastAPI):
    """Setup OpenTelemetry instrumentation matching Flask behavior."""
    
    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls="ping,health,livez,readyz,stats",  # Exclude health checks
        tracer_provider=trace.get_tracer_provider(),
    )
    
    # Add custom span processor if needed
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    
    if isinstance(trace.get_tracer_provider(), TracerProvider):
        # Already configured, skip
        pass
```

### OTel Configuration Parity

| Environment Variable | Flask | FastAPI | Notes |
|---------------------|-------|---------|-------|
| `OTEL_SERVICE_NAME` | `drum` | `drum` | Must match |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Configured | Same | |
| `OTEL_TRACES_SAMPLER` | `parentbased_traceidratio` | Same | |
| `OTEL_TRACES_SAMPLER_ARG` | `1.0` | Same | |

---

## Prometheus Metrics

### Metrics Exposed

If using Prometheus middleware:

```python
# In middleware.py
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
REQUEST_COUNT = Counter(
    "drum_requests_total",
    "Total request count",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "drum_request_duration_seconds",
    "Request latency",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

PREDICTIONS_IN_PROGRESS = Gauge(
    "drum_predictions_in_progress",
    "Number of predictions currently being processed"
)

# Memory metrics
MEMORY_USAGE = Gauge(
    "drum_memory_bytes",
    "Current memory usage in bytes",
    ["type"]  # rss, vms, shared
)

class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        method = request.method
        endpoint = self._normalize_endpoint(request.url.path)
        
        PREDICTIONS_IN_PROGRESS.inc()
        start_time = time.perf_counter()
        
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            status = 500
            raise
        finally:
            duration = time.perf_counter() - start_time
            REQUEST_COUNT.labels(method, endpoint, status).inc()
            REQUEST_LATENCY.labels(method, endpoint).observe(duration)
            PREDICTIONS_IN_PROGRESS.dec()
        
        return response
    
    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint for metric labels."""
        # Avoid high cardinality
        if path.startswith("/predict"):
            return "/predict"
        if path.startswith("/transform"):
            return "/transform"
        return path.rstrip("/") or "/"
```

### Metrics Endpoint

```python
# In routes.py
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
```

---

## Health Check Endpoints

### Kubernetes-Compatible Health Checks

| Endpoint | Purpose | Flask | FastAPI |
|----------|---------|-------|---------|
| `/ping` | Legacy liveness | ✅ | ✅ |
| `/health/` | Legacy readiness | ✅ | ✅ |
| `/livez` | K8s liveness probe | ❌ (add) | ✅ |
| `/readyz` | K8s readiness probe | ❌ (add) | ✅ |

```python
# In routes.py
@router.get("/livez")
async def liveness():
    """Kubernetes liveness probe."""
    return {"status": "alive"}

@router.get("/readyz")
async def readiness():
    """Kubernetes readiness probe."""
    # Check model is loaded and ready
    if not app.state.worker_ctx._running:
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "reason": "worker not running"}
        )
    
    if hasattr(predictor, "readiness_probe"):
        result = predictor.readiness_probe()
        if result.get("status") != "ok":
            return JSONResponse(status_code=503, content=result)
    
    return {"status": "ready"}
```

---

## Structured Logging (JSON)

For production environments, structured JSON logging is recommended:

```python
# In run_uvicorn.py
import logging
import json
from datetime import datetime

class JSONLogFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""
    
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, "request_id"):
            log_record["request_id"] = record.request_id
        if hasattr(record, "user_id"):
            log_record["user_id"] = record.user_id
        
        return json.dumps(log_record)

def configure_structured_logging():
    """Configure structured JSON logging if enabled."""
    from datarobot_drum import RuntimeParameters
    
    if RuntimeParameters.has("DRUM_JSON_LOGGING"):
        if str(RuntimeParameters.get("DRUM_JSON_LOGGING")).lower() in ("true", "1"):
            handler = logging.StreamHandler()
            handler.setFormatter(JSONLogFormatter())
            
            # Apply to all drum loggers
            for name in ["drum", "uvicorn.access", "uvicorn.error"]:
                logger = logging.getLogger(name)
                logger.handlers = [handler]
```

---

## Observability Parity Checklist

Before switching to FastAPI:

- [ ] Access log format matches or is parseable by existing tools
- [ ] All `/stats/` keys present and correctly calculated
- [ ] OTel span names identical
- [ ] OTel span attributes identical
- [ ] Prometheus metrics (if used) have same names and labels
- [ ] Health check endpoints return same status codes
- [ ] Structured logging (if used) has same schema
- [ ] Log levels configurable via same environment variables
