# Plan: custom_model_runner/datarobot_drum/drum/enum.py

Add FastAPI extension file name constant.

## Changes:
- Add `FASTAPI_EXT_FILE_NAME = "custom_fastapi"` (parallel to `FLASK_EXT_FILE_NAME = "custom_flask"`).

```python
# Existing:
FLASK_EXT_FILE_NAME = "custom_flask"

# Add new:
FASTAPI_EXT_FILE_NAME = "custom_fastapi"
```

This constant will be used by `load_fastapi_extensions()` in `fastapi/extensions.py` to locate custom FastAPI hook files.
