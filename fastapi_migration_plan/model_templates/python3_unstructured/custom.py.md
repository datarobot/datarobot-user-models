# Plan: model_templates/python3_unstructured/custom.py

Update template to be compatible with FastAPI by removing `werkzeug` dependency.

## Overview

The `score_unstructured` function in the template uses `werkzeug.formparser.parse_form_data`. This should be updated to a more generic way of handling multipart data so it works with both Flask and FastAPI.

## Changes:

### 1. Imports
```python
# Remove
import werkzeug
```

### 2. Update `score_unstructured`

```python
def score_unstructured(model, data, query, **kwargs):
    # ...
    if headers and "multipart/form-data" in headers.get("Content-Type"):
        # Replace werkzeug-based multipart parsing with a more compatible approach
        # ...
```

## Notes:
- Since this is a template, we want to provide code that is easy for users to understand and maintain across different DRUM versions.
