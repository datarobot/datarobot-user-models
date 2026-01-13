# Plan: tests/functional/test_parity_flask_fastapi.py

Functional tests to ensure response parity between Flask and FastAPI servers.

## Overview

This test suite runs the same set of requests against both Flask/Gunicorn and FastAPI/Uvicorn servers and compares the responses (status code, body, headers).

## Proposed Implementation

```python
"""
Copyright 2024 DataRobot, Inc. and its affiliates.
All rights reserved.
"""
import pytest
import requests
import os
from datarobot_drum.drum.root_predictors.drum_server_utils import DrumServerRun
from tests.constants import PYTHON, REGRESSION, BINARY, TESTS_DATA_PATH

@pytest.mark.parametrize("target_type", [REGRESSION, BINARY])
class TestServerParity:
    """Verify that Flask and FastAPI return identical results."""

    @pytest.fixture(scope="class")
    def model_dir(self, resources, tmp_path_factory, target_type):
        from datarobot_drum.drum.root_predictors.utils import _create_custom_model_dir
        tmp_dir = tmp_path_factory.mktemp(f"parity_{target_type}")
        return _create_custom_model_dir(resources, tmp_dir, None, target_type, PYTHON)

    def test_predict_parity(self, resources, model_dir, target_type):
        test_file = os.path.join(TESTS_DATA_PATH, "juniors_3_year_stats_regression.csv")
        
        # 1. Get Flask Response
        os.environ["MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE"] = "flask"
        with DrumServerRun(
            resources.target_types(target_type),
            resources.class_labels(None, target_type),
            model_dir,
        ) as flask_run:
            with open(test_file, "rb") as f:
                flask_resp = requests.post(flask_run.url_server_address + "/predict/", files={"X": f})
        
        # 2. Get FastAPI Response
        os.environ["MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE"] = "fastapi"
        with DrumServerRun(
            resources.target_types(target_type),
            resources.class_labels(None, target_type),
            model_dir,
        ) as fastapi_run:
            with open(test_file, "rb") as f:
                fastapi_resp = requests.post(fastapi_run.url_server_address + "/predict/", files={"X": f})
        
        # 3. Compare
        assert flask_resp.status_code == fastapi_resp.status_code
        assert flask_resp.json() == fastapi_resp.json()
        # Verify specific headers that should be present in both
        for header in ["X_Request_ID", "X-Drum-Version"]:
            assert header in flask_resp.headers
            assert header in fastapi_resp.headers

    def test_info_parity(self, resources, model_dir, target_type):
        # 1. Flask
        os.environ["MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE"] = "flask"
        with DrumServerRun(..., model_dir) as flask_run:
            flask_info = requests.get(flask_run.url_server_address + "/info/").json()

        # 2. FastAPI
        os.environ["MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE"] = "fastapi"
        with DrumServerRun(..., model_dir) as fastapi_run:
            fastapi_info = requests.get(fastapi_run.url_server_address + "/info/").json()

        # Compare (except for 'drumServer' field which will differ)
        flask_info.pop("drumServer")
        fastapi_info.pop("drumServer")
        assert flask_info == fastapi_info
```
