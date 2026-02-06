# Plan: tests/functional/test_streaming_responses.py

Functional tests for streaming responses (SSE, chunked) and proxying via `/directAccess/`.

## Overview

FastAPI handles streaming differently than Flask. These tests ensure that:
1.  `/chat/completions` correctly streams tokens (SSE).
2.  `/directAccess/` correctly proxies streaming content from backends (NIM/OpenAI).
3.  Large payloads are handled without timing out or OOMing.

## Proposed Implementation

```python
import pytest
import requests
import json
from datarobot_drum.drum.root_predictors.drum_server_utils import DrumServerRun

class TestStreaming:
    def test_chat_streaming_parity(self, resources, llm_model_dir):
        """Verify SSE streaming for chat completions."""
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True
        }
        
        # Test FastAPI
        os.environ["MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE"] = "fastapi"
        with DrumServerRun(..., llm_model_dir) as fastapi_run:
            resp = requests.post(
                fastapi_run.url_server_address + "/v1/chat/completions",
                json=payload,
                stream=True
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["Content-Type"]
            
            # Verify we can read chunks
            chunks = []
            for line in resp.iter_lines():
                if line:
                    chunks.append(line.decode("utf-8"))
            assert len(chunks) > 0
            assert chunks[0].startswith("data: ")

    def test_direct_access_proxy_streaming(self, resources, nim_model_dir, mock_nim_backend):
        """Verify that /directAccess/ proxies streaming from a backend."""
        # Setup mock backend to stream 100MB
        mock_nim_backend.set_stream_response(size_mb=100)
        
        os.environ["MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE"] = "fastapi"
        with DrumServerRun(..., nim_model_dir) as fastapi_run:
            resp = requests.get(
                fastapi_run.url_server_address + "/directAccess/large-file",
                stream=True
            )
            assert resp.status_code == 200
            
            bytes_received = 0
            for chunk in resp.iter_content(chunk_size=1024*1024):
                bytes_received += len(chunk)
            
            assert bytes_received == 100 * 1024 * 1024
```
