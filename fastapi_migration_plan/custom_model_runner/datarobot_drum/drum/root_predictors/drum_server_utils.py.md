# Plan: custom_model_runner/datarobot_drum/drum/root_predictors/drum_server_utils.py

Update `DrumServerRun` to support FastAPI testing.

## Overview

The `DrumServerRun` class is a helper used in functional tests to start and manage a DRUM server process. It needs to be updated to support the new `fastapi` server type.

## Proposed Implementation

### 1. Update `__init__` signature

```python
def __init__(
    self,
    target_type,
    class_labels,
    code_dir,
    # ... other existing params ...
    server_type: Optional[str] = None,
):
    # ... existing init code ...
    self._server_type = server_type or os.environ.get("MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE", "flask")
```

### 2. Update `get_command()` method

```python
def get_command(self):
    """
    Build the command to start the DRUM server.
    """
    cmd = [
        sys.executable,
        "-m",
        "datarobot_drum.drum.entry_point",
        "server",
        "--code-dir", self._code_dir,
        "--address", self._address,
    ]
    
    # Add target type and class labels
    if self._target_type:
        cmd.extend(["--target-type", self._target_type])
    # ... add other params ...
    
    return cmd
```

### 3. Update environment setup in `__enter__`

```python
def __enter__(self):
    """
    Start the DRUM server process.
    """
    env = os.environ.copy()
    
    # Set the server type environment variable
    if self._server_type:
        env["MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE"] = self._server_type
        logger.info("Starting DRUM server with type: %s", self._server_type)
    
    # ... existing subprocess launch code ...
    self._proc = subprocess.Popen(
        self.get_command(),
        env=env,
        # ... rest of Popen params ...
    )
    
    # Wait for server to be ready
    self._wait_for_server()
    
    return self
```

### 4. Wait for server logic

FastAPI/Uvicorn might take slightly longer to start than the Flask dev server due to async loop initialization and lifespan events. We use a 30-second timeout by default, which is sufficient for most models. The `_wait_for_server` method now explicitly checks if the process is still alive while waiting to fail fast if the server crashes on startup.

```python
def _wait_for_server(self, timeout=30):
    """
    Wait until the server starts responding to /ping/.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # We use /ping/ as the canonical readiness check for all server types
            response = requests.get(f"http://{self._address}/ping/", timeout=1.0)
            if response.status_code == 200:
                logger.info("DRUM server is ready")
                return
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pass
        
        # Check if process crashed
        if self._proc.poll() is not None:
            stdout, stderr = self._proc.communicate()
            raise RuntimeError(
                f"DRUM server process exited unexpectedly with code {self._proc.returncode}.\n"
                f"STDOUT: {stdout}\nSTDERR: {stderr}"
            )
        
        time.sleep(0.5)
    
    raise TimeoutError(f"DRUM server failed to start within {timeout} seconds at {self._address}")
```

## Key Changes Summary:

| Feature | Change |
|---------|--------|
| `server_type` param | Added to `__init__` to allow explicit server selection in tests |
| Environment Variable | Sets `MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE` before launching process |
| Command | Remains the same as it uses `entry_point.py` which handles selection |
| Reliability | Robust `_wait_for_server` to handle different startup times |

## Usage in Tests:

```python
# Test with FastAPI
with DrumServerRun(..., server_type="fastapi") as run:
    response = requests.post(run.url_server_address + "/predict/", ...)

# Test with Gunicorn
with DrumServerRun(..., server_type="gunicorn") as run:
    # ...
```

## Notes:
- Default behavior is preserved (defaults to "flask" or environment variable).
- This enables easy parametrization of existing functional tests.
- The `DrumServerRun` instance manages the lifecycle of the subprocess correctly for all server types.
