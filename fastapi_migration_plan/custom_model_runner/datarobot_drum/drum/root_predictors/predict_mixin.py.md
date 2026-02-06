# Plan: custom_model_runner/datarobot_drum/drum/root_predictors/predict_mixin.py

Update `PredictMixin` to support both Flask and FastAPI request/response handling.

## Overview

The `PredictMixin` class contains methods (`do_predict_structured`, `do_transform`, `do_predict_unstructured`, `do_chat`) that currently use Flask's `request` object directly. These need to be updated to work with both Flask and FastAPI request objects using a `RequestAdapter`.

## Request Adapter Utility

This class provides a unified interface for accessing request data regardless of the framework.

```python
from typing import Union, Optional, Any, Dict
import json

class RequestAdapter:
    """Adapter for unified access to Flask and FastAPI requests."""
    
    def __init__(self, request):
        self._request = request
        self._is_fastapi = hasattr(request, 'state')  # FastAPI Request has state
    
    @property
    def is_fastapi(self) -> bool:
        return self._is_fastapi
    
    @property
    def content_type(self) -> Optional[str]:
        if self._is_fastapi:
            return self._request.headers.get("content-type")
        return self._request.content_type
    
    @property
    def headers(self) -> Dict[str, str]:
        return dict(self._request.headers)
    
    def get_data(self) -> bytes:
        """Get raw request body."""
        if self._is_fastapi:
            # FastAPI body is pre-fetched in endpoint handler
            return getattr(self._request.state, 'body', b'')
        return self._request.get_data()
    
    def get_file(self, key: str = "X") -> Optional[Dict]:
        """Get uploaded file from request."""
        if self._is_fastapi:
            files = getattr(self._request.state, 'files', {})
            # DRUM typically expects the input file under key "X"
            return files.get(key)
        
        # Flask
        file_obj = self._request.files.get(key)
        if file_obj:
            return {
                "content": file_obj.read(),
                "filename": file_obj.filename,
                "content_type": file_obj.content_type
            }
        return None
    
    @property
    def query_params(self) -> Dict[str, str]:
        if self._is_fastapi:
            return dict(self._request.query_params)
        return self._request.args.to_dict()
```

## Changes to PredictMixin Methods:

### 1. Unified Request Handling
All `do_*` methods will start by wrapping the request:
```python
def do_predict_structured(self, logger, request=None):
    if request is None:
        from flask import request as flask_request
        request = flask_request
    
    adapter = RequestAdapter(request)
    # ... use adapter ...
```

### 2. Update `do_chat` for SSE Streaming
```python
def do_chat(self, logger=None, request=None, is_fastapi: bool = False):
    """Handle chat completion request with SSE support."""
    if request is None:
        from flask import request as flask_request
        request = flask_request
    
    adapter = RequestAdapter(request)
    
    # Get request body
    body_data = adapter.get_data()
    completion_create_params = json.loads(body_data)
    
    headers = adapter.headers
    result = self._predictor.chat(completion_create_params, headers=headers)
    
    from datarobot_drum.drum.root_predictors.predict_mixin import is_streaming_response
    
    if not is_streaming_response(result):
        response = result.to_dict()
        return response, 200
    else:
        # Streaming response (SSE)
        if is_fastapi:
            from fastapi.responses import StreamingResponse
            return StreamingResponse(
                self._stream_openai_chunks(result),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
            ), 200
        else:
            from flask import Response, stream_with_context
            return Response(
                stream_with_context(self._stream_openai_chunks(result)),
                mimetype="text/event-stream",
            ), 200

def _stream_openai_chunks(self, stream):
    """Generator for SSE format."""
    for chunk in stream:
        chunk_json = chunk.to_json(indent=None)
        for line in chunk_json.splitlines():
            yield f"data: {line}\n"
        yield "\n"
    yield "data: [DONE]\n\n"
```

### 3. Framework-Agnostic File Fetching
```python
@staticmethod
def _fetch_data_from_request(file_key, logger=None, request=None):
    """Fetch data from request supporting both Flask and FastAPI."""
    if request is None:
        from flask import request as flask_request
        request = flask_request
    
    adapter = RequestAdapter(request)
    
    # Try to get file from files collection
    file_info = adapter.get_file(file_key)
    
    if file_info is not None:
        binary_data = file_info["content"]
        mimetype = StructuredInputReadUtils.resolve_mimetype_by_filename(file_info["filename"])
    else:
        # Try raw body
        binary_data = adapter.get_data()
        if len(binary_data) > 0:
            mimetype, _ = PredictMixin._validate_content_type_header(adapter.content_type)
        else:
            raise ValueError(f"No data found for key {file_key}")
    
    return binary_data, mimetype, None
```

## Key Differences Summary:

| Feature | Flask Implementation | FastAPI Implementation |
|---------|----------------------|-----------------------|
| Request Body | `request.get_data()` | `request.state.body` |
| File Upload | `request.files` | `request.state.files` |
| Streaming | `stream_with_context` | `StreamingResponse` |
| SSE Header | `mimetype="text/event-stream"` | `media_type="text/event-stream"` |

## Notes:
- `RequestAdapter` abstracts the differences in request object access.
- For FastAPI, we rely on the endpoint pre-fetching async data into `request.state`.
- Chat completions support both blocking and SSE streaming responses for both frameworks.
- File handling supports both multipart form-data and raw body uploads.
