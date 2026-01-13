# Plan: tests/functional/test_drum_server_custom_flask.py Removal/Migration

Migration of Flask-specific functional tests to FastAPI.

## Overview

The `tests/functional/test_drum_server_custom_flask.py` file contains tests that specifically check for Flask backend behavior. These need to be migrated to test the FastAPI backend.

## Required Changes

### 1. Update assertions
Update tests that assert the backend is "flask":
```python
# Line 73
assert response.json()["drumServer"] == "flask"
```
**Change:** Assert `fastapi` instead.

### 2. Rename or Create Parallel Test
Consider creating `tests/functional/test_drum_server_custom_fastapi.py` or updating the existing test to be backend-agnostic if possible.

## Implementation Details
- Ensure all authentication passthrough and error cases are tested against the FastAPI server.
- Review and migrate all test cases to `tests/functional/test_drum_server_fastapi.py` or similar.
