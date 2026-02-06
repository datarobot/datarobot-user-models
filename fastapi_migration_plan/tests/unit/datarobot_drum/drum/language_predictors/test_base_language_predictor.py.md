# Plan: tests/unit/datarobot_drum/drum/language_predictors/test_base_language_predictor.py

Update unit tests to remove `werkzeug.exceptions.BadRequest` dependency.

## Overview

The tests currently use `werkzeug.exceptions.BadRequest`. These should be updated to use a more generic exception or `pytest.raises` with appropriate status code checks if needed.

## Changes:

### 1. Imports
```python
# Replace
from werkzeug.exceptions import BadRequest

# With
from fastapi import HTTPException # or a custom DRUM exception
```

### 2. Update test cases

Update `test_hook_wrong_response_type`, `test_failing_hook_with_mlops`, and `test_failing_in_middle_of_stream` to use the new exception type.
