# Plan: custom_model_runner/datarobot_drum/drum/root_predictors/transform_helpers.py

Replace `werkzeug.formparser.parse_form_data` with a framework-agnostic implementation or FastAPI-compatible logic.

## Overview

The `parse_multi_part_response` function uses `werkzeug.formparser.parse_form_data` which is tied to the WSGI/Flask ecosystem. This needs to be updated to support FastAPI (ASGI) while maintaining compatibility.

## Changes:

### 1. Imports
```python
# Replace
from werkzeug.formparser import parse_form_data

# With (example using python-multipart or manual parsing if needed)
# Or handle it within the function depending on the input type
```

### 2. Update `parse_multi_part_response`

```python
def parse_multi_part_response(response):
    # If response is from FastAPI/httpx, it might already have parsed data
    # or we use a more generic multipart parser.
    
    content_type = response.headers.get("Content-Type", "")
    if "multipart/form-data" not in content_type:
        return {}

    # New implementation using a more generic approach
    # ...
```

## Notes:
- Since DRUM will support both Flask and FastAPI, this helper should be able to handle both cases or be called differently depending on the context.
- Consider using `python-multipart` which is already a FastAPI dependency.
