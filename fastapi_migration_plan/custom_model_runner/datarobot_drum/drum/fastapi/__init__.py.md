# Plan: custom_model_runner/datarobot_drum/drum/fastapi/__init__.py

Package initialization for the FastAPI integration.

## Proposed Implementation:

```python
"""
FastAPI integration package for DRUM.
"""
from datarobot_drum.drum.fastapi.app import app, create_app
from datarobot_drum.drum.fastapi.context import FastAPIWorkerCtx, create_ctx
from datarobot_drum.drum.fastapi.config import UvicornConfig

__all__ = [
    "app",
    "create_app",
    "FastAPIWorkerCtx",
    "create_ctx",
    "UvicornConfig",
]
```
