# Plan: tests/conftest.py

Shared pytest fixtures for FastAPI migration tests.

## Proposed Implementation

```python
"""
Copyright 2024 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import pytest
import os
from pathlib import Path

from tests.constants import TESTS_ROOT_PATH

@pytest.fixture(scope="session")
def resources():
    """
    Fixture to provide access to test resources and helper methods.
    (Existing fixture from main conftest.py)
    """
    from tests.utils import TestResources
    return TestResources(TESTS_ROOT_PATH)

@pytest.fixture(autouse=True)
def cleanup_env():
    """
    Ensure DRUM environment variables are cleaned up between tests.
    """
    yield
    for key in list(os.environ.keys()):
        if key.startswith("MLOPS_RUNTIME_PARAM_DRUM_") or key == "URL_PREFIX":
            os.environ.pop(key, None)
```
