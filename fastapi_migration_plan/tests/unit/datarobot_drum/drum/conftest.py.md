# Plan: tests/unit/datarobot_drum/drum/conftest.py

Update unit test fixtures to support both Flask and FastAPI testing.

## Overview

The `conftest.py` file provides central fixtures for unit tests. It currently contains Flask-specific fixtures that need to be updated or complemented with FastAPI/httpx equivalents.

## Current Flask-Specific Code

```python
from datarobot_drum.drum.server import create_flask_app
from httpx import WSGITransport

@pytest.fixture
def test_flask_app():
    with patch("datarobot_drum.drum.server.create_flask_app") as mock_create_flask_app, patch(
        "datarobot_drum.drum.root_predictors.prediction_server.PredictionServer._run_flask_app"
    ):
        app = create_flask_app()
        app.config.update({"TESTING": True})
        mock_create_flask_app.return_value = app
        yield app

@pytest.fixture
def openai_client(test_flask_app):
    return OpenAI(
        base_url="http://localhost:8080",
        api_key="<KEY>",
        http_client=httpx.Client(transport=WSGITransport(app=test_flask_app)),
    )
```

## Required Changes

### 1. Add FastAPI/httpx Fixtures

```python
from fastapi.testclient import TestClient
from datarobot_drum.drum.fastapi.app import create_fastapi_app

@pytest.fixture
def test_fastapi_app(prediction_server):
    """Create a test FastAPI application using the prediction server's predictor."""
    app = create_fastapi_app(prediction_server._predictor)
    return app

@pytest.fixture
def fastapi_client(test_fastapi_app):
    """Test client for FastAPI app."""
    return TestClient(test_fastapi_app)

@pytest.fixture
def openai_fastapi_client(test_fastapi_app):
    """OpenAI client pointing to the FastAPI test application."""
    from httpx import ASGITransport
    return OpenAI(
        base_url="http://localhost:8080",
        api_key="<KEY>",
        http_client=httpx.Client(transport=ASGITransport(app=test_fastapi_app)),
    )
```

### 2. Generalize `openai_client`

Update `openai_client` to be able to use either Flask or FastAPI backend for testing parity.

```python
@pytest.fixture
def openai_client(request, test_flask_app, test_fastapi_app):
    server_type = request.config.getoption("--server-type", default="flask")
    if server_type == "fastapi":
        from httpx import ASGITransport
        return OpenAI(
            base_url="http://localhost:8080",
            api_key="<KEY>",
            http_client=httpx.Client(transport=ASGITransport(app=test_fastapi_app)),
        )
    else:
        from httpx import WSGITransport
        return OpenAI(
            base_url="http://localhost:8080",
            api_key="<KEY>",
            http_client=httpx.Client(transport=WSGITransport(app=test_flask_app)),
        )
```

## Notes

- Maintain `test_flask_app` for backward compatibility.
- Ensure `ASGITransport` is used for FastAPI and `WSGITransport` for Flask.
- The `prediction_server` fixture might need updates to ensure its predictor is properly initialized for both cases.
