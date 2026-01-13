# Removal Plan: custom_model_runner/datarobot_drum/drum/entry_point.py

Remove Gunicorn entry point.

## Actions:
- Remove `from datarobot_drum.drum.gunicorn.run_gunicorn import main as main_gunicorn`.
- Remove the check `if os.environ.get(ArgumentOptionsEnvVars.DRUM_SERVER_TYPE) == "gunicorn"`.
- Remove `main_gunicorn()` call.
