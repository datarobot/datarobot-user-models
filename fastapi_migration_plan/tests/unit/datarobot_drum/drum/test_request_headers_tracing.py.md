# Plan: tests/unit/datarobot_drum/drum/test_request_headers_tracing.py

Update request header tracing unit tests to cover both Flask and FastAPI server implementations.

## Overview

The `test_request_headers_tracing.py` file verifies that custom headers (like `X-DataRobot-Consumer-Id`) are correctly captured in OpenTelemetry spans. It currently uses Flask's `test_client`.

## Current Flask-Specific Code

```python
@pytest.fixture
def prediction_client(test_flask_app, prediction_server):
    return test_flask_app.test_client()

@pytest.mark.usefixtures("patch_otel_context", "patch_do_predict_unstructured")
def test_predict_unstructured_span_includes_consumer_headers(prediction_client, span_store):
    headers = {
        "X-DataRobot-Consumer-Id": "abc123",
        "X-DataRobot-Consumer-Type": "external",
        "Content-Type": "application/json",
    }
    resp = prediction_client.post("/predictUnstructured/", data=json.dumps({}), headers=headers)
    assert resp.status_code == 200
```

## Required Changes

### 1. Add FastAPI Test Client Fixture

```python
@pytest.fixture
def fastapi_prediction_client(fastapi_client, prediction_server):
    """Test client for FastAPI-based header tracing tests."""
    return fastapi_client
```

### 2. Parametrize Tracing Tests

```python
@pytest.mark.parametrize("server_type", ["flask", "fastapi"])
def test_predict_unstructured_span_includes_consumer_headers(
    server_type, request, span_store, patch_otel_context, patch_do_predict_unstructured
):
    if server_type == "flask":
        client = request.getfixturevalue("prediction_client")
    else:
        client = request.getfixturevalue("fastapi_prediction_client")
        
    headers = {
        "X-DataRobot-Consumer-Id": "abc123",
        "X-DataRobot-Consumer-Type": "external",
        "Content-Type": "application/json",
    }
    
    # URL might slightly differ or be the same depending on FastAPI route definitions
    url = "/predictUnstructured/"
    
    resp = client.post(url, data=json.dumps({}), headers=headers)
    assert resp.status_code == 200
    # ... verify spans ...
```

### 3. Verify Specific FastAPI Routes

Ensure all traced routes in FastAPI are tested:
- `/predictUnstructured/`
- `/v1/chat/completions`
- `/nim/some/path` (Direct access)
- `/transform/`
- `/invocations`

## Notes

- FastAPI uses different middleware/dependency injection for header extraction compared to Flask's `@before_request` or similar.
- Testing both ensures the OpenTelemetry instrumentation is correctly applied to both framework-specific request handlers.
- Ensure `test_fastapi_app` fixture from updated `conftest.py` is available.
