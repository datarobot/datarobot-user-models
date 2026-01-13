# Strategy: Removing Flask and Werkzeug

Steps to completely decommission Flask from DRUM after FastAPI migration is stable.

## Phase 1: Cleanup
- Remove `custom_model_runner/datarobot_drum/drum/gunicorn/` directory.
- Remove `create_flask_app`, `get_flask_app`, and all `Blueprint` logic from `server.py`.
- Remove Flask-specific error handlers.

## Phase 2: Refactoring
- In `prediction_server.py`, remove all conditional checks `if isinstance(self.app, Flask)`.
- Make `FastAPI` the only supported app type.
- Remove `TimeoutWSGIRequestHandler` as it's Flask/Werkzeug specific.
- Remove Flask-specific logic from `StdoutFlusher` and `ResourceMonitor`.
    - `ResourceMonitor` currently uses `flask.request` for some metadata; transition to a framework-agnostic `RequestState` or pass required info explicitly.
    - Ensure `StdoutFlusher` activity tracking works via FastAPI middleware instead of `after_request` hooks.

## Phase 3: Dependencies
- Remove the following from `requirements.txt`:
    - `flask`
    - `gunicorn`
    - `gevent`
    - `werkzeug` (usually comes with flask)

## Phase 4: Test Cleanup
- Remove `tests/functional/test_drum_server_custom_flask.py`.
- Remove `tests/fixtures/custom_flask_demo_auth.py`.
- Update `DrumServerRun` to default to `fastapi` and remove `flask` option.

## Phase 5: Environment & Docker Updates
- Update `docker/dropin_env_base/Dockerfile` (and other base images: `dropin_env_base_jdk`, `dropin_env_base_r`, etc.):
    - Install `fastapi`, `uvicorn[standard]`, `httpx`.
    - (Optional) Remove `gunicorn`, `gevent`, `flask` if not needed for backward compatibility.
    - Ensure Python 3.8+ is available.
- Update `example_dropin_environments/` to include FastAPI-based examples or update existing ones.
- Audit all `public_dropin_environments` for Flask usage and provide migration path for each.
- Ensure minimum Python version in base images is 3.8+ as required by FastAPI/Uvicorn.

## Timeline & Dual-Support Period

| Stage | Duration | Activities |
|-------|----------|------------|
| **Beta (Current)** | 4-6 weeks | FastAPI available via `DRUM_SERVER_TYPE=fastapi`. Both Flask and FastAPI are supported. |
| **Release Candidate** | 2-4 weeks | FastAPI becomes the default server for new environments. Flask remains for backward compatibility. |
| **LTS Support** | 6 months | Flask-based server is deprecated but supported. Users are encouraged to migrate. |
| **Decommissioning** | End of support | Flask, Gunicorn, and Gevent are removed from base images and DRUM codebase. |

During the **LTS Support** period, all new features and performance optimizations will be prioritized for the FastAPI server. Flask-based server will only receive critical security updates.
