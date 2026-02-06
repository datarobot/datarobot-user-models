# Plan: custom_model_runner/datarobot_drum/drum/entry_point.py

Support for launching the FastAPI/Uvicorn server alongside existing Flask/Gunicorn options.

## Overview

The entry point module is responsible for selecting and launching the appropriate server type based on runtime parameters. Currently it supports Flask (development) and Gunicorn (production). We need to add FastAPI/Uvicorn support.

## Current Implementation

```python
from datarobot_drum.drum.gunicorn.run_gunicorn import main_gunicorn
from datarobot_drum.drum.main import main
from datarobot_drum import RuntimeParameters
from datarobot_drum.drum.utils.setup import setup_options

from datarobot_drum.drum.enum import ArgumentsOptions


def run_drum_server():
    options = setup_options()
    if (
        options.subparser_name == ArgumentsOptions.SERVER
        and RuntimeParameters.has("DRUM_SERVER_TYPE")
        and str(RuntimeParameters.get("DRUM_SERVER_TYPE")).lower() == "gunicorn"
    ):
        main_gunicorn()
    else:
        main()


if __name__ == "__main__":
    run_drum_server()
```

## Proposed Implementation

```python
"""
Copyright 2024 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
from datarobot_drum.drum.main import main
from datarobot_drum import RuntimeParameters
from datarobot_drum.drum.utils.setup import setup_options
from datarobot_drum.drum.enum import ArgumentsOptions


def _get_server_type() -> str:
    """
    Determine the server type from RuntimeParameters.
    Auto-detects based on extension files if not explicitly set.
    
    Returns:
        Server type string: "flask" (default), "gunicorn", or "fastapi"
    """
    if not RuntimeParameters.has("DRUM_SERVER_TYPE"):
        # Auto-fallback logic
        from pathlib import Path
        code_dir = RuntimeParameters.get("__custom_model_path__", ".")
        has_flask = (Path(code_dir) / "custom_flask.py").exists()
        has_fastapi = (Path(code_dir) / "custom_fastapi.py").exists()
        
        if has_flask and not has_fastapi:
            return "flask"
        
        return "flask"  # Default unchanged for M1
    
    server_type = str(RuntimeParameters.get("DRUM_SERVER_TYPE")).lower()
    
    # Validate server type
    valid_types = {"flask", "gunicorn", "fastapi", "uvicorn"}
    if server_type not in valid_types:
        raise ValueError(
            f"Invalid DRUM_SERVER_TYPE: '{server_type}'. "
            f"Valid options are: {', '.join(sorted(valid_types))}"
        )
    
    # Normalize uvicorn to fastapi
    if server_type == "uvicorn":
        server_type = "fastapi"
    
    return server_type


def run_drum_server():
    """
    Main entry point for DRUM server.
    
    Selects and launches the appropriate server based on DRUM_SERVER_TYPE:
    - "flask" (default): Development Flask server via main()
    - "gunicorn": Production Gunicorn server with Flask app
    - "fastapi" or "uvicorn": Production Uvicorn server with FastAPI app
    """
    options = setup_options()
    
    # Only check server type for SERVER subcommand
    if options.subparser_name != ArgumentsOptions.SERVER:
        main()
        return
    
    server_type = _get_server_type()
    
    if server_type == "gunicorn":
        from datarobot_drum.drum.gunicorn.run_gunicorn import main_gunicorn
        main_gunicorn()
    
    elif server_type == "fastapi":
        from datarobot_drum.drum.fastapi.run_uvicorn import main_uvicorn
        main_uvicorn()
    
    else:  # flask (default)
        main()


if __name__ == "__main__":
    run_drum_server()
```

## Key Changes

| Aspect | Before | After |
|--------|--------|-------|
| Server types | flask, gunicorn | flask, gunicorn, fastapi |
| Import | Eager import of gunicorn | Lazy import of all servers |
| Validation | None | Validates DRUM_SERVER_TYPE value |
| Aliases | None | "uvicorn" → "fastapi" |

## Server Type Selection Flow

```
┌─────────────────────────────────────┐
│          run_drum_server()          │
└─────────────────────────────────────┘
                   │
                   ▼
        ┌──────────────────┐
        │ subparser_name   │
        │ == SERVER ?      │
        └──────────────────┘
               │ no         │ yes
               ▼            ▼
           main()    ┌─────────────────┐
                     │ DRUM_SERVER_TYPE│
                     └─────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   "flask"             "gunicorn"          "fastapi"
        │                   │                   │
        ▼                   ▼                   ▼
    main()           main_gunicorn()     main_uvicorn()
```

## Environment Variables

| Variable | Description | Values |
|----------|-------------|--------|
| `MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE` | Server type selection | `flask`, `gunicorn`, `fastapi`, `uvicorn` |

The runtime parameter `DRUM_SERVER_TYPE` is read via `RuntimeParameters.get()` which checks for:
- `MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE` environment variable
- Or the value in the runtime parameters file

## Backward Compatibility

- Default behavior unchanged: Flask development server
- Existing `DRUM_SERVER_TYPE=gunicorn` continues to work
- No changes to CLI arguments required

## Error Handling

Invalid server type values will raise `ValueError` with a helpful message:

```
ValueError: Invalid DRUM_SERVER_TYPE: 'invalid'. Valid options are: fastapi, flask, gunicorn, uvicorn
```

## Notes

- Imports are lazy to avoid loading unnecessary dependencies
- The "uvicorn" alias is provided for clarity but internally maps to "fastapi"
- Server selection only applies to the SERVER subcommand; other commands always use `main()`
