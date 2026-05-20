# Copyright 2026 DataRobot, Inc.
#
# Minimal sandbox runner image used by DataRobotWorkloadSandbox.
#
# Ships a standalone runner script (no datarobot-genai install needed
# inside the container) plus the data libraries user code commonly needs.

FROM python:3.12-slim

RUN pip install --no-cache-dir \
    "polars>=1.0.0,<2.0.0" \
    "pyarrow>=21.0.0,<22.0.0" \
    "datarobot>=3.10.0,<4.0.0" \
    "requests>=2.32.4,<3.0.0" \
    "httpx>=0.28.1,<1.0.0"

RUN mkdir -p /opt/dr_sandbox && chown 65534:65534 /opt/dr_sandbox
COPY runner.py /opt/dr_sandbox/runner.py

USER 65534:65534
WORKDIR /opt/dr_sandbox

CMD ["python", "/opt/dr_sandbox/runner.py"]
