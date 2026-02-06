# Removal Plan: tests/functional/test_drum_server_custom_flask.py

Removal of Flask-specific tests.

## Actions:
- Delete `tests/functional/test_drum_server_custom_flask.py`.
- Ensure all its test cases are covered by `test_drum_server_fastapi.py`.
