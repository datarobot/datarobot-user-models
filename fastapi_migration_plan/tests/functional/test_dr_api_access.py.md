# Plan: tests/functional/test_dr_api_access.py

Tests for DataRobot API access parity when running with FastAPI.

## Overview

This test verifies that the custom model can access the DataRobot API when permitted. It uses a mock Flask-based web server to simulate the DataRobot API. Since Flask is being removed as a dependency, this mock server needs to be migrated to FastAPI or another lightweight alternative.

## Proposed Implementation

```python
"""
Copyright 2024 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import contextlib
import json
import multiprocessing
import os
import socket
import pytest
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from retry import retry

from datarobot_drum.drum.enum import ArgumentsOptions
from datarobot_drum.drum.root_predictors.utils import _create_custom_model_dir
from datarobot_drum.drum.root_predictors.utils import _exec_shell_cmd
from tests.constants import PYTHON_UNSTRUCTURED_DR_API_ACCESS, UNSTRUCTURED
from tests.functional.utils import SimpleCache

class TestDrApiAccessFastAPI:
    """Contains cases to test DataRobot API access with FastAPI server."""

    WEBSERVER_HOST = "localhost"
    API_TOKEN = "zzz123"

    @contextlib.contextmanager
    def local_webserver_stub(self, webserver_port, expected_version_queries=1):
        init_cache_data = {"actual_ping_queries": 0, "actual_version_queries": 0, "token": ""}
        with SimpleCache(init_cache_data) as cache:
            app = FastAPI()

            def _extract_token_from_header(request: Request):
                auth_header = request.headers.get("Authorization")
                return auth_header.replace("Token ", "") if auth_header else ""

            @app.get("/ping/")
            async def ping(request: Request):
                cache.inc_value("actual_ping_queries")
                return {"response": "pong", "token": _extract_token_from_header(request)}

            @app.get("/api/v2/version/")
            async def version(request: Request):
                saved_content = cache.read_cache()
                saved_content["token"] = _extract_token_from_header(request)
                saved_content["actual_version_queries"] += 1
                cache.save_cache(saved_content)
                return {"major": 2, "minor": 28, "versionString": "2.28.0"}

            def run_server():
                uvicorn.run(app, host=self.WEBSERVER_HOST, port=webserver_port, log_level="error")

            proc = multiprocessing.Process(target=run_server)
            proc.start()

            try:
                yield

                @retry((AssertionError,), delay=1, tries=5)
                def _verify_expected_queries():
                    cache_data = cache.read_cache()
                    assert cache_data["token"] == self.API_TOKEN
                    assert cache_data["actual_version_queries"] == 1 + expected_version_queries

                _verify_expected_queries()
            finally:
                proc.terminate()
                proc.join()

    # ... rest of the test methods similar to original test_dr_api_access.py ...
```
