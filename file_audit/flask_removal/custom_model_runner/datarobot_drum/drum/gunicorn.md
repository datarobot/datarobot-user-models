# Removal Plan: custom_model_runner/datarobot_drum/drum/gunicorn/

Complete removal of Gunicorn and WSGI support.

## Actions:
- Delete the entire directory `custom_model_runner/datarobot_drum/drum/gunicorn/`.
- This includes:
    - `__init__.py`
    - `app.py`
    - `context.py`
    - `gunicorn.conf.py`
    - `run_gunicorn.py`
