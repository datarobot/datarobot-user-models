# Removal Plan: custom_model_runner/datarobot_drum/drum/runtime.py

Update `DrumRuntime` to support both Flask and FastAPI, and handle error server selection.

## Current State

The file currently has heavy Flask dependencies:

```python
from flask import Flask
from datarobot_drum.drum.server import get_flask_app
```

## Actions

### Phase 1: Add FastAPI Support (Dual Mode)

1. **Update imports**:
```python
from typing import Optional, Union, TYPE_CHECKING
if TYPE_CHECKING:
    from flask import Flask
    from fastapi import FastAPI
```

2. **Rename `flask_app` to `app`**:
```python
class DrumRuntime:
    def __init__(self, app: Optional[Union["Flask", "FastAPI"]] = None):
        # ... 
        self.app = app  # Renamed from flask_app
```

3. **Update `__exit__` for error server selection**:
```python
def __exit__(self, exc_type, exc_value, exc_traceback):
    # ... existing validation checks ...
    
    from datarobot_drum import RuntimeParameters
    server_type = "flask"
    if RuntimeParameters.has("DRUM_SERVER_TYPE"):
        server_type = str(RuntimeParameters.get("DRUM_SERVER_TYPE")).lower()

    if server_type in ["fastapi", "uvicorn"]:
        from datarobot_drum.drum.fastapi.error_server import run_error_server_fastapi
        run_error_server_fastapi(host, port, exc_value)
    else:
        run_error_server(host, port, exc_value, self.app)
```

### Phase 2: Flask Removal

1. Remove Flask imports entirely
2. Remove `run_error_server` Flask implementation
3. Make `run_error_server_fastapi` the default

## Notes

- `run_error_server_fastapi` is imported lazily to avoid dependencies if not used
- The `RunMode.SERVER` check is already framework-agnostic
- Error server displays initialization errors to users when model loading fails
