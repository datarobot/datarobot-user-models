# Plan: tests/unit/datarobot_drum/drum/test_model_templates.py

Update model template unit tests to cover both Flask and FastAPI server implementations.

## Overview

The `test_model_templates.py` file tests score and chat hooks from model templates using the prediction server and OpenAI client. These tests currently depend on the Flask implementation via the `test_flask_app` fixture (indirectly through `openai_client`).

## Current Flask-Specific Logic

```python
@pytest.mark.usefixtures("prediction_server")
@pytest.mark.parametrize("is_streaming", [True, False])
def test_dummy_chat_chat(openai_client, chat_python_model_adapter, is_streaming):
    # This test uses openai_client which is currently tied to Flask via WSGITransport
```

## Required Changes

### 1. Update Test to Support Multiple Backends

The test should be updated to run against both Flask and FastAPI backends to ensure template compatibility.

```python
@pytest.mark.parametrize("server_type", ["flask", "fastapi"])
@pytest.mark.parametrize("is_streaming", [True, False])
def test_dummy_chat_chat(server_type, request, chat_python_model_adapter, is_streaming):
    """Test the 'python3 dummy chat' hook across different server implementations."""
    
    # Get the appropriate client based on server_type
    if server_type == "flask":
        client = request.getfixturevalue("openai_client")
    else:
        client = request.getfixturevalue("openai_fastapi_client")
        
    chat_python_model_adapter.chat_hook = dummy_chat_chat
    prompt = "Tell me a story"

    completion = client.chat.completions.create(
        model=CHAT_COMPLETIONS_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=is_streaming,
    )
    # ... rest of the test ...
```

### 2. Add FastAPI Marker

```python
@pytest.mark.fastapi
def test_dummy_chat_chat_fastapi(openai_fastapi_client, chat_python_model_adapter):
    # Dedicated FastAPI test if needed
```

## Notes

- The primary goal is to ensure that model templates (hooks) work identically regardless of the underlying web framework (Flask or FastAPI).
- Testing both backends is crucial for detecting framework-specific issues in hook execution or data marshalling.
- Ensure that the `openai_fastapi_client` (to be added in `conftest.py`) is used for FastAPI-based tests.
