# Plan: tests/unit/datarobot_drum/drum/test_prediction_server.py

Update unit tests to cover both Flask and FastAPI server implementations.

## Overview

The `test_prediction_server.py` file contains Flask-specific tests that need to be updated or supplemented with FastAPI equivalents.

## Current Flask-Specific Tests

```python
def test_run_flask_app(processes_param, expected_processes, request_timeout):
    # Tests Flask app configuration

def test_request_id_in_flask_app(test_flask_app):
    # Tests request ID handling in Flask

def test_drum_version_in_flask_app(test_flask_app):
    # Tests DRUM version header in Flask
```

## Required Changes

### 1. Add FastAPI Test Fixtures

```python
import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def test_fastapi_app():
    """Create a test FastAPI application."""
    from datarobot_drum.drum.fastapi.app import create_fastapi_app
    from unittest.mock import Mock
    
    predictor = Mock()
    predictor.predict.return_value = {"predictions": [1, 2, 3]}
    
    app = create_fastapi_app(predictor)
    return TestClient(app)
```

### 2. Add Parallel FastAPI Tests

```python
def test_run_uvicorn_app(workers_param, expected_workers):
    """Test Uvicorn app configuration."""
    params = {
        "host": "localhost",
        "port": "6789",
        "run_language": "python",
        "target_type": "regression",
        "deployment_config": None,
    }
    if workers_param:
        params["workers"] = workers_param

    with patch.object(PredictionServer, "_setup_predictor"):
        server = PredictionServer(params)

    # Verify Uvicorn configuration
    config = server._get_uvicorn_config()
    assert config.workers == expected_workers


def test_request_id_in_fastapi_app(test_fastapi_app):
    """Test request ID handling in FastAPI."""
    response = test_fastapi_app.get(
        "/predict/",
        headers={"X-Request-Id": "test-request-id"}
    )
    assert response.headers.get("X-Request-Id") == "test-request-id"


def test_drum_version_in_fastapi_app(test_fastapi_app):
    """Test DRUM version header in FastAPI."""
    response = test_fastapi_app.get("/")
    assert "X-DRUM-Version" in response.headers
```

### 3. Parametrize Existing Tests

```python
@pytest.mark.parametrize("server_type", ["flask", "fastapi"])
def test_request_id_handling(server_type, test_flask_app, test_fastapi_app):
    """Test request ID handling across server types."""
    client = test_flask_app if server_type == "flask" else test_fastapi_app
    # ... test implementation
```

## Notes

- Keep Flask tests for backward compatibility during transition
- Add `@pytest.mark.fastapi` marker for FastAPI-specific tests
- Ensure test coverage parity between Flask and FastAPI implementations
- Update `conftest.py` with new fixtures for FastAPI testing
