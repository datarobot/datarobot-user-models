# Plan: custom_model_runner/datarobot_drum/drum/root_predictors/utils.py

Replace `werkzeug.http.parse_options_header` with a framework-agnostic implementation.

## Overview

The `get_mimetype_charset_from_content_type_header` function uses `werkzeug.http.parse_options_header`. This should be replaced to remove the Werkzeug dependency.

## Changes:

### 1. Imports
```python
# Remove
import werkzeug
```

### 2. Update `get_mimetype_charset_from_content_type_header`

```python
def get_mimetype_charset_from_content_type_header(header):
    # Use a standard library or more generic way to parse headers
    # Example using cgi (deprecated in 3.12) or better:
    if not header:
        return None, None
    
    parts = header.split(";")
    mimetype = parts[0].strip().lower()
    charset = None
    for part in parts[1:]:
        if "charset=" in part.lower():
            charset = part.split("=")[1].strip().strip('"')
    return mimetype, charset
```

## Notes:
- Python 3.12 deprecated `cgi.parse_header`, so a manual split or a modern library like `email.message.Message` could be used.
