# Strategy: Removing Flask and Werkzeug

Steps to completely decommission Flask from DRUM after FastAPI migration is stable.

## Phase 0: Prerequisites & Environment Audit
- **Python Version Audit**: Ensure all target environments and base images support Python 3.8+. FastAPI and Uvicorn require 3.8 or higher.
- **Dependency Audit**: Check for potential conflicts with existing user dependencies (e.g., `pydantic` versions).
- **Feature Flag Setup**: Implement `DRUM_FASTAPI_ENABLED` to allow controlled rollout.

## Phase 1: Cleanup
- Remove `custom_model_runner/datarobot_drum/drum/gunicorn/` directory.
- Remove `create_flask_app`, `get_flask_app`, and all `Blueprint` logic from `server.py`.
- Remove Flask-specific error handlers.

## Phase 2: Refactoring
- In `prediction_server.py`, remove all conditional checks `if isinstance(self.app, Flask)`.
- Make `FastAPI` the only supported app type.
- Remove `TimeoutWSGIRequestHandler` as it's Flask/Werkzeug specific.
- **Refactor `ResourceMonitor`**:
    - Remove imports of `flask.request`.
    - Ensure `collect_drum_info()` uses the tree-climbing logic to find the `entry_point` process, as Uvicorn's process structure can be deeper.
    - Fully transition to Cgroups v2 as the primary memory discovery mechanism.
- **Refactor `StdoutFlusher`**:
    - Remove the `after_request` integration.
    - Exclusively use `StdoutFlusherMiddleware` for activity tracking.
    - Ensure the flusher thread's lifecycle is strictly tied to `FastAPIWorkerCtx`.
- **Refactor `common.py`**:
    - Update `setup_otel` to detect multiprocessing in a server-agnostic way (replace Flask-specific comments/logic).
- **Refactor `PredictMixin`**:
    - Remove `flask.request` and `flask.Response`.
    - Make `RequestAdapter` the mandatory way to access request data.
    - Replace `stream_with_context` with FastAPI's `StreamingResponse`.

## Phase 3: Dependencies
- Remove the following from `requirements.txt`:
    - `flask`
    - `gunicorn`
    - `gevent`
    - `werkzeug` (usually comes with flask)

## Phase 4: Test Cleanup
- Remove `tests/functional/test_drum_server_custom_flask.py`.
- Remove `tests/fixtures/custom_flask_demo_auth.py`.
- **Refactor Unit Tests**:
    - Update `tests/unit/datarobot_drum/drum/conftest.py` with FastAPI fixtures.
    - Update `tests/unit/datarobot_drum/drum/test_model_templates.py` to support multiple backends.
    - Update `tests/unit/datarobot_drum/drum/test_request_headers_tracing.py` for FastAPI compatibility.
    - Update `tests/unit/datarobot_drum/drum/test_prediction_server.py`.
- Update `tests/functional/test_dr_api_access.py`:
    - Migrate mock DataRobot API server from Flask to FastAPI/uvicorn.
- Update `tests/functional/test_mlops_monitoring.py`:
    - Migrate mock MLOps API server from Flask to FastAPI/uvicorn.
- Update `tests/functional/test_inference.py` and `tests/functional/test_inference_per_framework.py`:
    - Replace `assert response_dict[ModelInfoKeys.DRUM_SERVER] == "flask"` with a check that supports both or defaults to `fastapi`.
- Update `tests/locust/README.md` to remove references to Flask and update to FastAPI.
- Update `DrumServerRun` to default to `fastapi` and remove `flask` option.

## Phase 5: Environment & Docker Updates
- Update `docker/dropin_env_base/Dockerfile` (and other base images: `dropin_env_base_jdk`, `dropin_env_base_r`, etc.):
    - Install `fastapi`, `uvicorn[standard]`, `httpx`.
    - (Optional) Remove `gunicorn`, `gevent`, `flask` if not needed for backward compatibility.
    - Ensure Python 3.8+ is available.
- Update `example_dropin_environments/` to include FastAPI-based examples or update existing ones.
- Audit all `public_dropin_environments` and `public_dropin_environments_sandbox` for Flask usage and provide migration path for each.
- Ensure minimum Python version in base images is 3.8+ as required by FastAPI/Uvicorn.

## Implementation Order & Recommendations

1. **`enum.py`**: Add constants and feature flags.
2. **`config.py`**: Implementation of Uvicorn configuration.
3. **`context.py`**: Define worker context and resource management.
4. **`server.py`**: Implement middleware and app factory.
5. **`extensions.py`**: Port extension loading logic.
6. **`app.py`**: Set up lifespan and ASGI entry point.
7. **`run_uvicorn.py`**: Create the server launcher.
8. **`entry_point.py`**: Integrate with DRUM CLI.
9. **`prediction_server.py`**: Final routing and initialization logic.
10. **Tests**: Verify parity and new functionality.

### Key Recommendations:
- **Feature Flags**: Use `DRUM_FASTAPI_ENABLED` for gradual rollout.
- **Monitoring**: Update `test_mlops_monitoring.py` to ensure metrics parity.
    - Specifically, verify that `MLOpsContext` and `PredictionServer` correctly report metrics when running under FastAPI.
    - Test both embedded and sidecar monitoring modes.
    - Ensure that async prediction calls in FastAPI are correctly timed and reported to MLOps.
- **Endpoints**: Ensure consistency between `/ping/` and `/ping` (trailing slash).
- **Errors**: Standardize error messages to match Flask format for client compatibility.
- **Health Checks**: Exclude health endpoints from OTel tracing to reduce noise.

## Timeline & Dual-Support Period

| Stage | Duration | Activities |
|-------|----------|------------|
| **Beta (Current)** | 4-6 weeks | FastAPI available via `DRUM_SERVER_TYPE=fastapi`. Both Flask and FastAPI are supported. |
| **Release Candidate** | 2-4 weeks | FastAPI becomes the default server for new environments. Flask remains for backward compatibility. |
| **LTS Support** | 6 months | Flask-based server is deprecated but supported. Users are encouraged to migrate. |
| **Decommissioning** | End of support | Flask, Gunicorn, and Gevent are removed from base images and DRUM codebase. |

During the **LTS Support** period, all new features and performance optimizations will be prioritized for the FastAPI server. Flask-based server will only receive critical security updates.
