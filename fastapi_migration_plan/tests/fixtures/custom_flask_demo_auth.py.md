# Plan: tests/fixtures/custom_flask_demo_auth.py Removal/Migration

Migration of Flask-specific test fixtures to FastAPI.

## Overview

The `tests/fixtures/custom_flask_demo_auth.py` file is a Flask-specific script used in functional tests to demonstrate custom authentication. It must be replaced by a FastAPI equivalent.

## Required Changes

### 1. Identify usages
This fixture is primarily used in `tests/functional/test_drum_server_custom_flask.py`.

### 2. Replace with FastAPI version
Create `tests/fixtures/custom_fastapi_demo_auth.py` (which already exists or is planned) and update tests to use it.

## Implementation Details
- This file is marked for removal in the Flask removal phase.
- During migration, ensure that `tests/fixtures/custom_fastapi_demo_auth.py` provides the same authentication logic but using FastAPI's `Depends` or middleware.
