#!/bin/bash

# Launch the FastAPI app on port 8080
exec uvicorn start_app:app --host 0.0.0.0 --port 8080 --app-dir /opt/code/
