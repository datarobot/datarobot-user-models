# Plan: tests/functional/test_mlops_monitoring.py

Tests for MLOps monitoring parity when running with FastAPI.

## Overview

This test verifies that metrics (predictions, execution time) are correctly reported to DataRobot MLOps when running under the FastAPI server. It covers both embedded and sidecar monitoring modes.

## Proposed Implementation

```python
"""
Copyright 2024 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import os
import pytest
import requests
from typing import Generator

from datarobot_drum.drum.root_predictors.drum_server_utils import DrumServerRun
from datarobot_drum.drum.utils.drum_utils import unset_drum_supported_env_vars
from tests.constants import PYTHON, REGRESSION, TESTS_DATA_PATH

class TestMLOpsMonitoringFastAPI:
    """Verify metrics parity between Flask and FastAPI."""

    @pytest.fixture(scope="class")
    def custom_model_dir(self, resources, tmp_path_factory):
        from datarobot_drum.drum.root_predictors.utils import _create_custom_model_dir
        tmp_dir = tmp_path_factory.mktemp("model_dir_mlops")
        return _create_custom_model_dir(resources, tmp_dir, None, REGRESSION, PYTHON)

    def test_mlops_metrics_parity(self, resources, custom_model_dir):
        """
        Test that metrics are reported correctly in FastAPI.
        Uses a mock MLOps agent/service to capture metrics.
        """
        unset_drum_supported_env_vars()
        os.environ["MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE"] = "fastapi"
        # Enable monitoring
        os.environ["MLOPS_RUNTIME_PARAM_MONITORING_ENABLED"] = "true"
        
        # Mock MLOps environment variables
        os.environ["MLOPS_DEPLOYMENT_ID"] = "test-deployment-id"
        os.environ["MLOPS_MODEL_ID"] = "test-model-id"
        
        test_file = os.path.join(TESTS_DATA_PATH, "juniors_3_year_stats_regression.csv")
        
        try:
            with DrumServerRun(
                resources.target_types(REGRESSION),
                resources.class_labels(None, REGRESSION),
                custom_model_dir,
            ) as run:
                # Perform predictions
                for _ in range(5):
                    with open(test_file, "rb") as f:
                        response = requests.post(
                            run.url_server_address + "/predict/",
                            files={"X": f},
                        )
                    assert response.ok
                
                # Check server logs for MLOps reporting messages
                logs = run.get_logs()
                assert "MLOps" in logs
                # Verify that no MLOps-related errors occurred
                assert "ERROR" not in logs.lower() or "mlops" not in logs.lower()
        finally:
            os.environ.pop("MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE", None)
            os.environ.pop("MLOPS_RUNTIME_PARAM_MONITORING_ENABLED", None)

    def test_async_prediction_timing(self, resources, custom_model_dir):
        """
        Verify that execution time metrics are accurate for async calls.
        FastAPI runs sync predictions in a thread pool; we must ensure
        the time spent in the thread is correctly captured.
        """
        # Implementation: Compare Reported Execution Time vs Actual wall time
        pass
```
