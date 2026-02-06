# Plan: tests/fixtures/unstructured_custom.py

Update test fixture to remove `werkzeug` dependency.

## Overview

The fixture uses `werkzeug` for multipart parsing. This should be replaced with a framework-agnostic implementation to support testing DRUM in FastAPI mode.

## Changes:

### 1. Imports
```python
# Remove
import werkzeug
```

### 2. Update `score_unstructured`

```python
def score_unstructured(model, data, query, **kwargs):
    # Update to generic multipart parsing
    # ...
```
