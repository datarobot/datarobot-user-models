# Plan: tests/functional/test_streaming_responses.py

Functional tests for streaming (SSE) and direct access proxying in FastAPI.

## Overview

This suite verifies that:
1. `/chat/completions` correctly streams responses using Server-Sent Events (SSE).
2. `/directAccess/` and `/nim/` correctly proxy streaming responses from backends using `httpx`.

## Proposed Implementation

```python
"""
Copyright 2024 DataRobot, Inc. and its affiliates.
All rights reserved.
"""
import pytest
import requests
import json
from datarobot_drum.drum.root_predictors.drum_server_utils import DrumServerRun
from tests.constants import PYTHON, TEXT_GENERATION

class TestStreamingFastAPI:
    """Verify streaming capabilities of the FastAPI server."""

    def test_chat_streaming_sse(self, resources, tmp_path):
        """Verify SSE streaming for chat endpoint."""
        # Use a dummy textgen model that supports streaming
        model_dir = resources.get_model_dir("python3_dummy_textgen")
        
        os.environ["MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE"] = "fastapi"
        with DrumServerRun(..., model_dir) as run:
            payload = {
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True
            }
            # Use stream=True in requests to capture chunks
            response = requests.post(
                run.url_server_address + "/v1/chat/completions",
                json=payload,
                stream=True
            )
            
            assert response.ok
            assert response.headers["Content-Type"] == "text/event-stream"
            
            chunks = []
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        content = decoded_line[6:]
                        if content == "[DONE]":
                            break
                        chunks.append(json.loads(content))
            
            assert len(chunks) > 0

    def test_direct_access_proxy_streaming(self, resources, tmp_path):
        """Verify that directAccess endpoint proxies streaming data correctly."""
        # This test requires a mock backend server or a specific model
        pass
```
