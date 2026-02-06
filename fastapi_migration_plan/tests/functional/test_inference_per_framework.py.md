# Plan: tests/functional/test_inference_per_framework.py Migration

Updates for `tests/functional/test_inference_per_framework.py` to support FastAPI.

## Overview

This test suite runs inference tests across multiple frameworks. Similar to `test_inference.py`, it contains hardcoded checks for the `"flask"` server type.

## Necessary Changes

### 1. Update `ModelInfoKeys.DRUM_SERVER` checks

Update the following assertion to be server-type aware:
```python
assert response_dict[ModelInfoKeys.DRUM_SERVER] == "flask"
```

### 2. Framework-specific parity

Ensure that framework-specific prediction hooks (e.g., for Scikit-Learn, XGBoost, etc.) continue to work identically under FastAPI. Pay special attention to:
- Large payload handling (CSV/MTX).
- Error response format consistency.
- Input data marshalling.

## Proposed Implementation (snippet)

```python
# Update server type check
expected_server = os.environ.get("DRUM_SERVER_TYPE", "flask")
assert response_dict[ModelInfoKeys.DRUM_SERVER] == expected_server
```
