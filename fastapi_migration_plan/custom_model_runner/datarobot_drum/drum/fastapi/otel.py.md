# Plan: custom_model_runner/datarobot_drum/drum/fastapi/otel.py

OpenTelemetry integration for FastAPI.

## Overview

FastAPI requires specific instrumentation for OpenTelemetry to capture request spans, headers, and metadata correctly. This module defines the instrumentation logic and middleware for FastAPI.

## Proposed Implementation

### 1. FastAPI Instrumentation

We will use `FastAPIInstrumentor` from `opentelemetry.instrumentation.fastapi`.

```python
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from datarobot_drum.drum.common import setup_otel

def instrument_app(app: FastAPI):
    """
    Instrument the FastAPI application with OpenTelemetry.
    Integrates with existing setup_otel() from common.py.
    """
    # Ensure base OTel setup is done
    setup_otel()
    
    # Define health endpoints to exclude from tracing
    excluded_urls = "/ping,/health,/,/info/,/stats/,/capabilities/"
    
    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls=excluded_urls,
    )
```

### 2. Custom Middleware for Span Attributes

To match the existing Flask instrumentation, we need to extract specific headers and attributes.

```python
from opentelemetry import trace
from datarobot_drum.drum.common import extract_request_headers

async def otel_middleware(request: Request, call_next):
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span(
        f"drum.{request.method} {request.url.path}",
        kind=trace.SpanKind.SERVER
    ) as span:
        # Add custom attributes from headers
        span.set_attributes(extract_request_headers(dict(request.headers)))
        
        response = await call_next(request)
        
        # Add response status code
        span.set_attribute("http.status_code", response.status_code)
        
        return response
```

### 3. Integration with `PredictionServer`

In `prediction_server.py`, during `_materialize_fastapi()`, we should apply the instrumentation.

```python
def _materialize_fastapi(self):
    # ... create app ...
    
    if RuntimeParameters.get("ENABLE_OTEL"):
        from datarobot_drum.drum.fastapi.otel import instrument_app
        instrument_app(app)
    
    # ...
```

## Key Differences from Flask

| Aspect | Flask | FastAPI |
|--------|-------|---------|
| Instrumentor | `FlaskInstrumentor` | `FastAPIInstrumentor` |
| Middleware | `before_request` / `after_request` | `BaseHTTPMiddleware` |
| Context Propagation | Automatic via `flask-otel` | Manual or via `FastAPIInstrumentor` |

## Notes:
- `FastAPIInstrumentor` handles context propagation (e.g., from `X-B3-TraceId` or W3C headers) automatically.
- We should ensure that `excluded_urls` matches the Flask configuration to avoid noise from health checks.
- The `extract_request_headers` utility from `common.py` should be used to maintain consistency in reported metadata.
