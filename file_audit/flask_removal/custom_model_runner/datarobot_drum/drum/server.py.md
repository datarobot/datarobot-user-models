# Removal Plan: custom_model_runner/datarobot_drum/drum/server.py

Cleanup Flask-related code and keep only FastAPI logic.

## Actions:
- Remove `import flask`, `from flask import Flask, Blueprint, request`.
- Delete `get_flask_app` function.
- Delete `base_api_blueprint` and `empty_api_blueprint`.
- Delete `before_request` and `after_request` (already replaced by FastAPI middlewares).
- Delete `create_flask_app`.
- Ensure common constants like `HTTP_200_OK`, `HEADER_REQUEST_ID` are preserved for FastAPI usage.
