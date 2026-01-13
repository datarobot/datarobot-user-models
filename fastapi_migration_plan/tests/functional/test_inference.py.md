# Plan: tests/functional/test_inference.py Migration

Updates for `tests/functional/test_inference.py` to support FastAPI and ensure parity.

## Overview

This test suite contains general inference tests. It currently has hardcoded checks for the `"flask"` server type in `ModelInfoKeys.DRUM_SERVER`.

## Necessary Changes

### 1. Update `ModelInfoKeys.DRUM_SERVER` checks

The tests currently assert that the server is Flask:
```python
assert response_dict[ModelInfoKeys.DRUM_SERVER] == "flask"
```

This needs to be updated to allow for `"fastapi"` when the `DRUM_SERVER_TYPE` environment variable is set to `"fastapi"`, or the check should be made server-agnostic.

### 2. Verify `/info/` endpoint parity

Ensure that the `/info/` endpoint in FastAPI returns the same keys and expected values as Flask, with the exception of `drumServer` which will be `"fastapi"`.

### 3. Production Mode Test

Refactor `test_custom_models_with_drum_in_production_mode`. In Flask, this meant running with multiple processes. In FastAPI, this corresponds to running with multiple Uvicorn workers. Ensure the test correctly verifies this behavior for both.

## Proposed Implementation (snippet)

```python
expected_server = os.environ.get("DRUM_SERVER_TYPE", "flask")
assert response_dict[ModelInfoKeys.DRUM_SERVER] == expected_server
```
