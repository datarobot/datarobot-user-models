# FastAPI Migration: Milestones

This document describes the division of work into milestones with clear boundaries and a list of files for each milestone.

## Related Documentation

- [SECURITY_AUDIT.md](SECURITY_AUDIT.md) - Security checklist for each milestone
- [COMPATIBILITY_MATRIX.md](COMPATIBILITY_MATRIX.md) - Python/Pydantic version support
- [OBSERVABILITY_MIGRATION.md](OBSERVABILITY_MIGRATION.md) - Logs and metrics parity
- [CANARY_DEPLOYMENT.md](CANARY_DEPLOYMENT.md) - Progressive rollout strategy
- [ROLLBACK_PLAN.md](ROLLBACK_PLAN.md) - Rollback procedures

---

## Milestones Overview

```mermaid
flowchart LR
    M0[M0: Security Audit] --> M1[M1: Core Infrastructure]
    M1 --> M2[M2: Dual-mode]
    M2 --> M3[M3: Stability]
    M3 --> M4[M4: Environments]
    M4 --> M5[M5: Canary Rollout]
    M5 --> M6[M6: Default Switch]
```

| Milestone | Name | Goal | Dependencies | Duration (Realistic) | Buffer | Risk Factor |
|------|----------|------|-------------|-------------------------|--------|-------------|
| **M0** | Security Audit | Audit dependencies, document security requirements | - | 1.5 weeks | +0.5 week | Low |
| **M1** | Core FastAPI Infrastructure | FastAPI/Uvicorn parallel to Flask/Gunicorn | M0 | 5 - 6 weeks | +2 weeks | Medium |
| **M2** | Dual-mode & Routing Separation | Support both servers, extract routes | M1 | 5 - 6 weeks | +2 weeks | High |
| **M3** | Stability & Resource Management | Tests, back-pressure, memory limits, observability parity | M2 | 6 - 8 weeks | +2 weeks | High |
| **M4** | Docker & Environments | Update Docker images and all environments | M3 | 4 weeks | +1 week | Medium |
| **M5** | Canary Rollout | Progressive traffic migration (1% → 100%) | M4 | 8 - 12 weeks | +4 weeks | Critical |
| **M6** | Default Switch | FastAPI by default, deprecation of Flask | M5 | 3 weeks | +1 week | Medium |

**Total Realistic Estimate: 33 - 44 weeks (8 - 11 months)**

> ⚠️ **Note:** Previous estimate of 23-28 weeks was optimistic. This revised estimate includes:
> - Integration issues with customer ML libraries (Pydantic conflicts, etc.)
> - Extended canary rollout for production stability
> - Time for customer migration support
> - Unexpected edge cases in production traffic

### Timeline Assumptions

- **2 developers** (1 senior + 1 mid-level) for M1-M3
- 1 senior developer for M4-M6
- Dedicated QA for M3 and M5
- No major interruptions
- Code review cycles included
- Buffer for unexpected issues
- Customer support load during M5

### Risk Mitigation

| Phase | Primary Risk | Mitigation |
|-------|-------------|------------|
| M1-M2 | Pydantic v1/v2 conflicts | Start with FastAPI <0.100, migrate to v2 later |
| M3 | ThreadPoolExecutor deadlocks | Implement backpressure early, load test extensively |
| M5 | Production regressions | Extended canary phases, automatic rollback triggers |
| M6 | Customer extension breakage | 3-month deprecation period, migration support |

---

## Milestone 0: Security Audit (NEW)

**Goal:** Audit all new dependencies and establish security baseline.

**Duration:** 1 week

### Deliverables

- [ ] Run `pip-audit` on FastAPI, Uvicorn, httpx
- [ ] Document CVE history for new dependencies
- [ ] Create [SECURITY_AUDIT.md](SECURITY_AUDIT.md) checklist
- [ ] Review SSL/TLS cipher configuration
- [ ] Document CORS security implications

### Exit Criteria

- No HIGH/CRITICAL CVEs in new dependencies
- Security checklist approved by security team

---

## Milestone 1: Core FastAPI Infrastructure

**Goal:** Add FastAPI/Uvicorn to `custom_model_runner` parallel to the existing Flask/Gunicorn without changing the current default behavior.

**Completion Criteria:** 
- `DRUM_SERVER_TYPE=fastapi` starts Uvicorn server
- Flask continues to work by default
- Automatic fallback to Flask if only `custom_flask.py` is present
- All existing tests pass
- Basic back-pressure (Semaphore) and Request Timeout middleware implemented

### Files to Modify (MODIFY)

| File | Change Description | Specification from migration plan |
|------|-------------------|-------------------------------|
| `custom_model_runner/requirements.txt` | Add fastapi, uvicorn[standard], httpx | [requirements.txt.md](custom_model_runner/requirements.txt.md) |
| `custom_model_runner/datarobot_drum/drum/enum.py` | Add FASTAPI_EXT_FILE_NAME and configuration constants | [enum.py.md](custom_model_runner/datarobot_drum/drum/enum.py.md) |
| `custom_model_runner/datarobot_drum/drum/entry_point.py` | Add server selection via _get_server_type() | [entry_point.py.md](custom_model_runner/datarobot_drum/drum/entry_point.py.md) |
| `custom_model_runner/datarobot_drum/drum/server.py` | Add create_fastapi_app() and get_fastapi_app() | [server.py.md](custom_model_runner/datarobot_drum/drum/server.py.md) |

### Files to Create (CREATE)

All files are created in `custom_model_runner/datarobot_drum/drum/fastapi/`:

| File | Description | Specification from migration plan |
|------|----------|-------------------------------|
| `__init__.py` | Module exports | [__init__.py.md](custom_model_runner/datarobot_drum/drum/fastapi/__init__.py.md) |
| `config.py` | UvicornConfig dataclass for parsing RuntimeParameters | [config.py.md](custom_model_runner/datarobot_drum/drum/fastapi/config.py.md) |
| `context.py` | FastAPIWorkerCtx - worker lifecycle management | [context.py.md](custom_model_runner/datarobot_drum/drum/fastapi/context.py.md) |
| `app.py` | FastAPI app with lifespan events (startup/shutdown) | [app.py.md](custom_model_runner/datarobot_drum/drum/fastapi/app.py.md) |
| `run_uvicorn.py` | Launcher for Uvicorn (analogous to run_gunicorn.py) | [run_uvicorn.py.md](custom_model_runner/datarobot_drum/drum/fastapi/run_uvicorn.py.md) |
| `otel.py` | OpenTelemetry middleware for FastAPI | [otel.py.md](custom_model_runner/datarobot_drum/drum/fastapi/otel.py.md) |
| `error_server.py` | Error server on FastAPI (analogous to Flask error server) | [error_server.py.md](custom_model_runner/datarobot_drum/drum/fastapi/error_server.py.md) |
| `routes.py` | Core route definitions (abstracted from logic) | [routes.py.md](custom_model_runner/datarobot_drum/drum/fastapi/routes.py.md) |
| `middleware.py` | Custom middlewares (Timeout, Semaphore, etc.) | [middleware.py.md](custom_model_runner/datarobot_drum/drum/fastapi/middleware.py.md) |

### Structure after M1

```
custom_model_runner/datarobot_drum/drum/
├── fastapi/                    # NEW MODULE
│   ├── __init__.py            # CREATE
│   ├── config.py              # CREATE - UvicornConfig
│   ├── context.py             # CREATE - FastAPIWorkerCtx
│   ├── app.py                 # CREATE - ASGI entry point
│   ├── routes.py              # CREATE - Core routes
│   ├── middleware.py          # CREATE - Timeout/Semaphore
│   ├── run_uvicorn.py         # CREATE - Launcher
│   ├── otel.py                # CREATE - OTel middleware
│   └── error_server.py        # CREATE - Error server
├── gunicorn/                   # EXISTING - no changes
│   ├── __init__.py
│   ├── app.py
│   ├── context.py
│   ├── gunicorn.conf.py
│   └── run_gunicorn.py
├── entry_point.py              # MODIFY - add server type selection
├── server.py                   # MODIFY - add FastAPI app factory
├── enum.py                     # MODIFY - add constants
└── ...
```

### Details of Changes by File

#### 1. `requirements.txt` - Add dependencies

```diff
 flask
 jinja2>=3.0.0
+fastapi>=0.100.0
+uvicorn[standard]>=0.23.0
+httpx>=0.24.0
 memory_profiler<1.0.0
```

#### 2. `enum.py` - Add constants

```python
# Extension file names
FLASK_EXT_FILE_NAME = "custom_flask"  # existing
FASTAPI_EXT_FILE_NAME = "custom_fastapi"  # NEW

# Server type selection
DRUM_SERVER_TYPE = "DRUM_SERVER_TYPE"  # NEW

# FastAPI/Uvicorn configuration constants
DRUM_FASTAPI_EXECUTOR_WORKERS = "DRUM_FASTAPI_EXECUTOR_WORKERS"
DRUM_FASTAPI_MAX_UPLOAD_SIZE = "DRUM_FASTAPI_MAX_UPLOAD_SIZE"
DRUM_FASTAPI_ENABLE_DOCS = "DRUM_FASTAPI_ENABLE_DOCS"
DRUM_UVICORN_LOOP = "DRUM_UVICORN_LOOP"
DRUM_UVICORN_MAX_REQUESTS = "DRUM_UVICORN_MAX_REQUESTS"
DRUM_UVICORN_GRACEFUL_TIMEOUT = "DRUM_UVICORN_GRACEFUL_TIMEOUT"
DRUM_UVICORN_KEEP_ALIVE = "DRUM_UVICORN_KEEP_ALIVE"
DRUM_UVICORN_LOG_LEVEL = "DRUM_UVICORN_LOG_LEVEL"

# CORS Configuration
DRUM_CORS_ENABLED = "DRUM_CORS_ENABLED"
DRUM_CORS_ORIGINS = "DRUM_CORS_ORIGINS"

# SSL/TLS Configuration (shared with gunicorn)
DRUM_SSL_CERTFILE = "DRUM_SSL_CERTFILE"
DRUM_SSL_KEYFILE = "DRUM_SSL_KEYFILE"
DRUM_SSL_KEYFILE_PASSWORD = "DRUM_SSL_KEYFILE_PASSWORD"
```

#### 3. `entry_point.py` - Add server selection

```python
def _get_server_type() -> str:
    """Determine server type from RuntimeParameters."""
    if not RuntimeParameters.has("DRUM_SERVER_TYPE"):
        return "flask"  # Default unchanged
    
    server_type = str(RuntimeParameters.get("DRUM_SERVER_TYPE")).lower()
    valid_types = {"flask", "gunicorn", "fastapi", "uvicorn"}
    
    if server_type not in valid_types:
        raise ValueError(f"Invalid DRUM_SERVER_TYPE: '{server_type}'")
    
    # Normalize uvicorn -> fastapi
    if server_type == "uvicorn":
        server_type = "fastapi"
    
    return server_type


def run_drum_server():
    options = setup_options()
    
    if options.subparser_name != ArgumentsOptions.SERVER:
        main()
        return
    
    server_type = _get_server_type()
    
    if server_type == "gunicorn":
        from datarobot_drum.drum.gunicorn.run_gunicorn import main_gunicorn
        main_gunicorn()
    elif server_type == "fastapi":
        from datarobot_drum.drum.fastapi.run_uvicorn import main_uvicorn
        main_uvicorn()
    else:
        main()  # Flask default
```

#### 4. `server.py` - Add FastAPI app factory

```python
from typing import Optional
from fastapi import FastAPI, APIRouter

def create_fastapi_app() -> FastAPI:
    """Create a new FastAPI application instance."""
    from datarobot_drum import RuntimeParameters
    
    enable_docs = (
        RuntimeParameters.has("DRUM_FASTAPI_ENABLE_DOCS") and
        str(RuntimeParameters.get("DRUM_FASTAPI_ENABLE_DOCS")).lower() in ("true", "1", "yes")
    )
    
    app = FastAPI(
        title="DRUM Prediction Server",
        docs_url="/docs" if enable_docs else None,
        redoc_url="/redoc" if enable_docs else None,
        openapi_url="/openapi.json" if enable_docs else None,
    )
    
    # Add CORS if enabled
    if RuntimeParameters.has("DRUM_CORS_ENABLED"):
        if str(RuntimeParameters.get("DRUM_CORS_ENABLED")).lower() in ("true", "1", "yes"):
            from fastapi.middleware.cors import CORSMiddleware
            origins = ["*"]
            if RuntimeParameters.has("DRUM_CORS_ORIGINS"):
                origins = str(RuntimeParameters.get("DRUM_CORS_ORIGINS")).split(",")
            app.add_middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
    
    return app


def get_fastapi_app(router: APIRouter, app: Optional[FastAPI] = None) -> FastAPI:
    """Get or create FastAPI app and include router."""
    if app is None:
        app = create_fastapi_app()
    app.include_router(router)
    return app
```

### Components correspondence Flask/Gunicorn → FastAPI/Uvicorn

| Gunicorn/Flask | FastAPI/Uvicorn | File |
|----------------|-----------------|------|
| `gunicorn/run_gunicorn.py` | `fastapi/run_uvicorn.py` | Launcher |
| `gunicorn/gunicorn.conf.py` | `fastapi/config.py` | Configuration |
| `gunicorn/context.py` (WorkerCtx) | `fastapi/context.py` (FastAPIWorkerCtx) | Worker lifecycle |
| `gunicorn/app.py` | `fastapi/app.py` | App entry point |
| `get_flask_app()` | `create_fastapi_app()` | App factory |
| `base_api_blueprint()` | `APIRouter` | Routing |

---

## Milestone 2: Dual-mode Prediction Server

**Goal:** Update `prediction_server.py` to support both operating modes.

### Files to Modify

| File | Change Description | Specification |
|------|-------------------|--------------|
| `root_predictors/prediction_server.py` | Add `_materialize_fastapi()` | [prediction_server.py.md](custom_model_runner/datarobot_drum/drum/root_predictors/prediction_server.py.md) |
| `root_predictors/predict_mixin.py` | Add `RequestAdapter` for request abstraction | [predict_mixin.py.md](custom_model_runner/datarobot_drum/drum/root_predictors/predict_mixin.py.md) |
| `root_predictors/stdout_flusher.py` | Add `StdoutFlusherMiddleware` for FastAPI | [stdout_flusher.py.md](custom_model_runner/datarobot_drum/drum/root_predictors/stdout_flusher.py.md) |
| `resource_monitor.py` | Update for Uvicorn process tree | [resource_monitor.py.md](custom_model_runner/datarobot_drum/drum/resource_monitor.py.md) |

### Files to Create

| File | Description | Specification |
|------|----------|--------------|
| `fastapi/extensions.py` | Load `custom_fastapi.py` | [extensions.py.md](custom_model_runner/datarobot_drum/drum/fastapi/extensions.py.md) |

---

## Milestone 3: Stability & Resource Management

**Goal:** Full test coverage, support for user extensions, and ensuring parity in performance and observability.

**Duration:** 5 - 6 weeks (includes buffer for debugging)

### Key Tasks:

#### 3.1 Backpressure & Memory Management (Week 1-2)
- [ ] Implement `ProductionBackpressure` class with queue depth limiting
- [ ] Add `SpooledUploadFile` for memory-efficient large file handling
- [ ] Implement rejection with 503 when queue is full
- [ ] Add backpressure metrics to `/stats/` endpoint

#### 3.2 Observability Parity (Week 2-3)
- [ ] Verify `/stats/` response keys match Flask (see [OBSERVABILITY_MIGRATION.md](OBSERVABILITY_MIGRATION.md))
- [ ] Ensure OpenTelemetry spans and attributes are identical
- [ ] Configure access log format matching Gunicorn
- [ ] Add Prometheus metrics (if applicable)

#### 3.3 Testing (Week 3-5)
- [ ] Create parity tests with semantic comparison (deepdiff)
- [ ] Add chaos testing (kill worker mid-request)
- [ ] Load testing with production-scale data
- [ ] Security tests per [SECURITY_AUDIT.md](SECURITY_AUDIT.md)

#### 3.4 Performance Benchmarking (Week 5-6)
- [ ] Latency comparison (p50/p95/p99)
- [ ] Throughput comparison (RPS)
- [ ] Memory footprint comparison
- [ ] Document results in [PERFORMANCE_BENCHMARKING.md](PERFORMANCE_BENCHMARKING.md)

### Exit Criteria
- [ ] All parity tests pass
- [ ] P99 latency within 20% of Flask
- [ ] No memory leaks over 24h test
- [ ] Security audit checklist complete

### Files to Create (tests)

| File | Description |
|------|----------|
| `tests/unit/datarobot_drum/drum/fastapi/test_config.py` | Unit tests for UvicornConfig |
| `tests/unit/datarobot_drum/drum/fastapi/test_context.py` | Unit tests for FastAPIWorkerCtx |
| `tests/functional/test_drum_server_fastapi.py` | Functional tests for FastAPI |
| `tests/functional/test_custom_fastapi_extensions.py` | Tests for custom_fastapi.py |
| `tests/fixtures/custom_fastapi_demo_auth.py` | Fixture for testing extensions |

### Files to Modify (tests)

| File | Change Description |
|------|-------------------|
| `tests/functional/test_inference.py` | Add parametrization for FastAPI |
| `tests/functional/test_inference_per_framework.py` | Add FastAPI server type |
| `tests/unit/datarobot_drum/drum/conftest.py` | Add FastAPI fixtures |

---

## Milestone 4: Docker & Environments

**Goal:** Update Docker images and public environments to support FastAPI.

### Files to Modify

| File/Directory | Change Description |
|-----------------|-------------------|
| `docker/dropin_env_base/Dockerfile` | Add FastAPI deps |
| `docker/dropin_env_base_jdk/Dockerfile` | Add FastAPI deps |
| `docker/dropin_env_base_r/Dockerfile` | Add FastAPI deps |
| `public_dropin_environments/*/requirements.txt` | Add fastapi, uvicorn, httpx |

---

## Milestone 5: Canary Rollout (NEW)

**Goal:** Progressive traffic migration with monitoring and automatic rollback.

**Duration:** 4 - 6 weeks (depends on issue discovery)

See [CANARY_DEPLOYMENT.md](CANARY_DEPLOYMENT.md) for detailed strategy.

### Phase Schedule

| Week | Phase | Traffic % | Criteria to Proceed |
|------|-------|-----------|---------------------|
| 1 | Shadow | 0% | No errors in shadow requests |
| 2 | Canary | 1% → 5% | Error rate < 0.1%, latency < +20% |
| 3 | Canary | 10% | Stable for 48h |
| 4 | Expansion | 25% → 50% | Stable for 72h |
| 5 | Majority | 75% → 90% | No customer issues |
| 6 | Full | 100% | 1 week observation |

### Exit Criteria
- [ ] 100% traffic on FastAPI for 1 week
- [ ] No critical bugs
- [ ] Support team trained
- [ ] Runbooks updated

---

## Milestone 6: Default Switch & Deprecation

**Goal:** Make FastAPI the default server and deprecate Flask.

**Duration:** 2 weeks

### Files to Modify

| File | Change Description |
|------|-------------------|
| `entry_point.py` | Change default to "fastapi" |
| `main.py` | Add DeprecationWarning for Flask |

### Deprecation Timeline

| Version | Change |
|---------|--------|
| N | FastAPI becomes default |
| N+1 | Deprecation warning for Flask |
| N+2 | Flask support removed |

### Exit Criteria
- [ ] Default changed to FastAPI
- [ ] Deprecation warnings in place
- [ ] Documentation updated
- [ ] Migration guide published

---

## Milestone 1 Checklist

### Dependencies
- [ ] Add `fastapi>=0.100.0` to requirements.txt
- [ ] Add `uvicorn[standard]>=0.23.0` to requirements.txt
- [ ] Add `httpx>=0.24.0` to requirements.txt

### Constants (enum.py)
- [ ] Add `FASTAPI_EXT_FILE_NAME`
- [ ] Add `DRUM_SERVER_TYPE`
- [ ] Add `DRUM_FASTAPI_*` constants
- [ ] Add `DRUM_UVICORN_*` constants
- [ ] Add `DRUM_CORS_*` constants

### FastAPI module
- [ ] Create `fastapi/__init__.py`
- [ ] Create `fastapi/config.py` with UvicornConfig
- [ ] Create `fastapi/context.py` with FastAPIWorkerCtx and Semaphore
- [ ] Create `fastapi/app.py` with lifespan and app.state.worker_ctx
- [ ] Create `fastapi/routes.py`
- [ ] Create `fastapi/middleware.py` (TimeoutMiddleware)
- [ ] Create `fastapi/run_uvicorn.py`
- [ ] Create `fastapi/otel.py`
- [ ] Create `fastapi/error_server.py`

### Integration
- [ ] Update `entry_point.py` with _get_server_type() and auto-fallback
- [ ] Add `create_fastapi_app()` in server.py
- [ ] Add `get_fastapi_app()` in server.py

### Validation and Health Checks
- [ ] `/livez` and `/readyz` endpoints comply with K8s best practices
- [ ] `/ping` and `/health` maintain backward compatibility
- [ ] Running with `DRUM_SERVER_TYPE=flask` works (default)
- [ ] Automatic fallback to Flask if only `custom_flask.py` is present
- [ ] All existing unit/functional tests pass
- [ ] Request timeout middleware correctly handles `DRUM_CLIENT_REQUEST_TIMEOUT`
