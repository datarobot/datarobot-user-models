# Removal Plan: custom_model_runner/datarobot_drum/drum/root_predictors/prediction_server.py

Remove all Flask dependencies and framework-switching logic.

## Actions:
- Remove `from flask import Response, jsonify, request`.
- Remove `from werkzeug.exceptions import HTTPException`.
- Remove `from werkzeug.serving import WSGIRequestHandler`.
- Delete `TimeoutWSGIRequestHandler` class.
- Update `PredictionServer.__init__`:
    - Remove `flask_app` parameter.
    - Set `self.app` strictly to `FastAPI` instance.
- Update `materialize()`:
    - Remove Flask-based route registration.
    - Keep only FastAPI `APIRouter` logic.
- Delete `_run_flask_app` method.
- Update `load_flask_extensions` to be removed (replaced by `load_fastapi_extensions`).
