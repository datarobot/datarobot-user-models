# Plan: custom_model_runner/datarobot_drum/drum/language_predictors/java_predictor/java_predictor.py

Replace `werkzeug` dependency with framework-agnostic or FastAPI-compatible logic.

## Overview

The `JavaPredictor` class uses `werkzeug.datastructures.ImmutableMultiDict` to handle query parameters in `predict_unstructured`. This should be replaced to avoid a hard dependency on Werkzeug.

## Changes:

### 1. Imports
```python
# Replace
from werkzeug.datastructures import ImmutableMultiDict

# With
try:
    from starlette.datastructures import ImmutableMultiDict
except ImportError:
    # Fallback or use a simple dict check if Starlette is not available
    ImmutableMultiDict = None
```

### 2. Update `predict_unstructured` method

```python
def predict_unstructured(self, data, **kwargs):
    # ...
    query = kwargs.get(UnstructuredDtoKeys.QUERY, dict())
    # ...
    if isinstance(query, dict):
        query_dict = query
    elif ImmutableMultiDict and isinstance(query, ImmutableMultiDict):
        query_dict = query.to_dict()
    else:
        # Fallback for other mapping types
        query_dict = dict(query)
    # ...
```

## Notes:
- Using `starlette.datastructures.ImmutableMultiDict` is preferred as it's a dependency of FastAPI.
- Added a fallback to ensure it works even if Starlette is not present (e.g., in legacy Flask mode if we decide to remove Werkzeug entirely).
