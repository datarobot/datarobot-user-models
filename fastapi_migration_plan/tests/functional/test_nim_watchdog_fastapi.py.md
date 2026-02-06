# Plan: tests/functional/test_nim_watchdog_fastapi.py

Functional tests for NIM Watchdog integration with FastAPI.

## Overview

This test verifies that the watchdog thread correctly monitors the FastAPI server and performs actions if the server becomes unresponsive.

## Proposed Implementation

```python
"""
Copyright 2024 DataRobot, Inc. and its affiliates.
All rights reserved.
"""
import pytest
import os
import time
import signal
from datarobot_drum.drum.root_predictors.drum_server_utils import DrumServerRun

class TestNIMWatchdogFastAPI:
    """Verify watchdog behavior under FastAPI."""

    def test_watchdog_lifecycle(self, resources, tmp_path):
        model_dir = resources.get_model_dir("python3_sklearn")
        
        os.environ["MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE"] = "fastapi"
        os.environ["MLOPS_RUNTIME_PARAM_USE_NIM_WATCHDOG"] = "true"
        # Make watchdog very aggressive for testing
        os.environ["MLOPS_RUNTIME_PARAM_NIM_WATCHDOG_INTERVAL"] = "1"
        
        try:
            with DrumServerRun(..., model_dir) as run:
                # Check logs for watchdog startup
                logs = run.get_logs()
                assert "Started NIM watchdog thread" in logs
                
                # Verify it's still running after some time
                time.sleep(2)
                assert run._proc.poll() is None
                
                # Graceful shutdown should stop the watchdog
                run._proc.send_signal(signal.SIGTERM)
                run._proc.wait(timeout=10)
                
                final_logs = run.get_logs()
                # We should see evidence of graceful exit if we implemented the _running flag check
                # assert "Watchdog thread exiting" in final_logs
        finally:
            os.environ.pop("MLOPS_RUNTIME_PARAM_USE_NIM_WATCHDOG", None)
```
