# Plan: custom_model_runner/datarobot_drum/drum/fastapi/run_uvicorn.py

Launcher script for Uvicorn, mirroring `run_gunicorn.py`.

## Overview

This module provides the entry point for running the DRUM server with Uvicorn. It handles:
- Configuration parsing
- Environment setup
- Uvicorn process launch

## Proposed Implementation:

```python
"""
Uvicorn launcher for DRUM FastAPI server.
Mirrors run_gunicorn.py functionality.
"""
import logging
import subprocess
from pathlib import Path
import sys
import os
import shlex

from datarobot_drum.drum.enum import LOGGER_NAME_PREFIX
from datarobot_drum.drum.fastapi.config import UvicornConfig

logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + __name__)


def main_uvicorn():
    """
    Launch the Uvicorn server with DRUM FastAPI application.
    
    This function:
    1. Resolves the package directory containing app.py
    2. Exports CLI args to DRUM_UVICORN_DRUM_ARGS environment variable
    3. Sets up PYTHONPATH to include the fastapi module directory
    4. Launches Uvicorn with the configured parameters
    """
    # Resolve directory containing this script so we can always find app.py
    package_dir = Path(__file__).resolve().parent
    app_module = package_dir / "app.py"
    
    if not app_module.is_file():
        raise FileNotFoundError(f"FastAPI app module not found: {app_module}")
    
    # Export all provided CLI args (excluding script) into DRUM_UVICORN_DRUM_ARGS
    # This allows the app.py to access original DRUM arguments
    extra_args = sys.argv
    if extra_args:
        try:
            os.environ["DRUM_UVICORN_DRUM_ARGS"] = shlex.join(extra_args)
        except AttributeError:
            # Python < 3.8 compatibility
            os.environ["DRUM_UVICORN_DRUM_ARGS"] = " ".join(shlex.quote(a) for a in extra_args)
    
    # Setup PYTHONPATH to include the fastapi module directory
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    if current_pythonpath:
        env["PYTHONPATH"] = f"{package_dir}{os.pathsep}{current_pythonpath}"
    else:
        env["PYTHONPATH"] = str(package_dir)
    
    # Get configuration from RuntimeParameters
    config = UvicornConfig.from_runtime_params()
    
    # Build Uvicorn command
    # Use the uvicorn module explicitly to avoid issues with shadowed scripts
    uvicorn_command = [
        sys.executable,
        "-m",
        "uvicorn",
        "app:app",  # module:variable; app.py sits next to this script
    ] + config.to_cli_args()
    
    # Add factory flag if using lifespan
    # uvicorn_command.append("--factory")  # Uncomment if using app factory pattern
    
    logger.info("Starting Uvicorn server with command: %s", " ".join(uvicorn_command))
    logger.info("Uvicorn config: workers=%d, host=%s, port=%d", 
                config.workers, config.host, config.port)
    
    try:
        subprocess.run(uvicorn_command, env=env, check=True)
    except FileNotFoundError:
        logger.error("uvicorn module not found. Ensure it is installed.")
        raise
    except subprocess.CalledProcessError as e:
        logger.error("Uvicorn exited with non-zero status %s", e.returncode)
        raise


def main_uvicorn_programmatic():
    """
    Alternative launcher using uvicorn.run() directly.
    Useful for programmatic startup without subprocess.
    """
    import uvicorn
    from datarobot_drum.drum.fastapi.app import create_app
    
    config = UvicornConfig.from_runtime_params()
    
    logger.info("Starting Uvicorn server programmatically")
    logger.info("Uvicorn config: workers=%d, host=%s, port=%d", 
                config.workers, config.host, config.port)
    
    # Note: When using workers > 1, uvicorn.run requires a string import path
    if config.workers > 1:
        uvicorn.run(
            "datarobot_drum.drum.fastapi.app:app",
            **config.to_uvicorn_kwargs()
        )
    else:
        # Single worker can use app instance directly
        app = create_app()
        uvicorn.run(
            app,
            **config.to_uvicorn_kwargs()
        )


if __name__ == "__main__":
    main_uvicorn()
```

## Lifespan Events

The lifespan events (startup/shutdown) are handled in `app.py`, not in this module. This differs from gunicorn where `post_worker_init` and `worker_exit` are in `gunicorn.conf.py`.

### Startup (equivalent to `post_worker_init`):
- Restore `sys.argv` from `DRUM_UVICORN_DRUM_ARGS`
- Reset `MAX_WORKERS` to 1 for single-worker mode
- Create and start `WorkerCtx`

### Shutdown (equivalent to `worker_exit`):
- Call `ctx.stop()` for graceful shutdown
- Call `ctx.cleanup()` for resource cleanup

See `app.py.md` for lifespan implementation details.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DRUM_UVICORN_DRUM_ARGS` | Original DRUM CLI arguments, space-separated |
| `PYTHONPATH` | Extended to include the fastapi module directory |
| `ADDRESS` | Server bind address in format `host:port` |

## Notes:
- The subprocess approach is preferred as it provides better process isolation.
- The programmatic approach (`main_uvicorn_programmatic`) is provided as an alternative for testing or single-worker scenarios.
- Unlike gunicorn, Uvicorn handles signals (SIGTERM, SIGINT) internally for graceful shutdown.
