# Plan: custom_model_runner/datarobot_drum/drum/runtime.py

Update `DrumRuntime` to support both Flask and FastAPI, and handle error server selection correctly.

## Changes:

### 1. Imports and Type Hinting
```python
from typing import Optional, Union, TYPE_CHECKING
if TYPE_CHECKING:
    from flask import Flask
    from fastapi import FastAPI
```

### 2. Update `DrumRuntime.__init__`
```python
class DrumRuntime:
    def __init__(self, app: Optional[Union["Flask", "FastAPI"]] = None):
        self.initialization_succeeded = False
        self.options = None
        self.cm_runner = None
        # OTEL services
        self.trace_provider = None
        self.metric_provider = None
        self.log_provider = None
        self.app = app # Renamed from flask_app
```

### 3. Update `DrumRuntime.__exit__` to select error server
```python
    def __exit__(self, exc_type, exc_value, exc_traceback):
        # ... (keep existing validation checks) ...

        # start 'error server'
        host_port_list = self.options.address.split(":", 1)
        host = host_port_list[0]
        port = int(host_port_list[1]) if len(host_port_list) == 2 else None

        from datarobot_drum import RuntimeParameters
        server_type = "flask"
        if RuntimeParameters.has("DRUM_SERVER_TYPE"):
            server_type = str(RuntimeParameters.get("DRUM_SERVER_TYPE")).lower()

        with verbose_stdout(self.options.verbose):
            if server_type in ["fastapi", "uvicorn"]:
                from datarobot_drum.drum.fastapi.error_server import run_error_server_fastapi
                run_error_server_fastapi(host, port, exc_value)
            else:
                run_error_server(host, port, exc_value, self.app)

        return False  # propagate exception further
```

## Key Considerations:
- `self.flask_app` should be renamed to `self.app` for consistency with other components.
- The `RunMode.SERVER` check is already framework-agnostic.
- `run_error_server_fastapi` is imported lazily to avoid dependencies if not used.
