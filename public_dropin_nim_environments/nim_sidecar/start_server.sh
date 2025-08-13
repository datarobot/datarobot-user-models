#!/bin/sh
# Copyright 2024 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
echo "Starting Custom Model environment with NIM"
set -e

if [ "${ENABLE_CUSTOM_MODEL_RUNTIME_ENV_DUMP}" = 1 ]; then
    echo "Environment variables:"
    env
fi

echo
echo "Starting DRUM server..."
echo

#exec gunicorn app:app --workers=4 --bind=0.0.0.0:8080 --backlog=4 & OK 20 Workers 30 parallel request 8 mb
#exec gunicorn app:app --workers=4 --bind=0.0.0.0:8080 --backlog=32 & probably ok
#exec gunicorn app:app --workers=32 --bind=0.0.0.0:8080 --backlog=512 & #failed (stopped)
#exec gunicorn app:app --workers=32 --bind=0.0.0.0:8080 --backlog=256 & #failing...
exec gunicorn app:app  -k gthread --workers=32 --bind=0.0.0.0:8080 --backlog=512 --threads 4 --timeout 120 --max-requests 1000 --max-requests-jitter 100  & #works
#exec gunicorn app:app  -k gthread --workers=128 --bind=0.0.0.0:8080 --backlog=512 --threads 4 --timeout 120 & #failed
#exec gunicorn app:app2 --workers=1 --bind=0.0.0.0:8080 &

# Wait for both processes
wait