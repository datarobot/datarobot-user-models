# Removal Plan: custom_model_runner/datarobot_drum/drum/root_predictors/predict_mixin.py

Rewrite request handling to be framework-agnostic or FastAPI-specific.

## Actions:
- Remove `from flask import request, Response, stream_with_context`.
- Replace `request.files.get(file_key)` with FastAPI `Request.form()` or `UploadFile` logic.
- Replace `request.data` and `request.content_type` with FastAPI equivalents.
- Replace `Response(response, mimetype=...)` with `fastapi.responses.Response` or `JSONResponse`.
- Replace `stream_with_context` with FastAPI `StreamingResponse`.
