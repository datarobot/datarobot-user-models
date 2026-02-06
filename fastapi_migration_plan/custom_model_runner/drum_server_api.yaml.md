# Plan: custom_model_runner/drum_server_api.yaml Migration

Updates to OpenAPI documentation to reflect FastAPI backend.

## Overview

The `drum_server_api.yaml` file provides the OpenAPI specification for the DRUM prediction server. It contains examples and descriptions that refer to the Flask backend.

## Required Changes

### 1. Update `drumServer` example
The example response for the `/info/` route shows `drumServer: flask`:
```yaml
# Line 99
drumServer: flask
```
**Change:** Update the example to show `fastapi`.

### 2. General Description Review
Review all descriptions to ensure they don't explicitly mention Flask unless referring to historical/deprecated behavior.

## Implementation Details
- Update the `info/` endpoint example in `custom_model_runner/drum_server_api.yaml`.
- Ensure the `drumServer` property description is accurate for both backends if both are still supported, or updated if only FastAPI is supported.
