# Copyright 2026 DataRobot, Inc.
#
# Minimal sandbox runner image used by DataRobotWorkloadSandbox.
#
# Ships a standalone runner script (no datarobot-genai install needed
# inside the container) plus the data libraries user code commonly needs.
#
# ============================================================================
#   *** USER CODE INSIDE THIS IMAGE IS HARD-CAPPED AT 1 HOUR (3600s) ***
#
#   runner.py enforces a SIGALRM-based wall-clock on the user snippet. Code
#   still running after the cap is interrupted and the container exits with
#   code 124. Override via the DR_SANDBOX_TIMEOUT_SECS env var, set to 0 to
#   disable. See README.md (## Execution timeout) for details. Note: this is
#   defense-in-depth for accidental hangs only — the real wall-clock
#   enforcement is caller-side (DataRobotWorkloadSandbox / Workload API).
# ============================================================================

# This is a private chain-guard development image stored in DataRobot's private
# registry. Replace it with your own development chain-guard image if you build
# your own.
ARG BASE_ROOT_IMAGE=datarobot/mirror_chainguard_datarobot.com_python-fips:3.12-dev
FROM ${BASE_ROOT_IMAGE} AS build

# The chainguard base defaults to a non-root user, which can't write to /opt;
# switch to root so the venv can be created at ${VIRTUAL_ENV}.
USER root

ENV VIRTUAL_ENV=/opt/venv
# Use the venv's own interpreter for pip so packages land in ${VIRTUAL_ENV}
# (which the final stage copies). Running ``/usr/bin/python -m pip`` here would
# install into the base interpreter's site-packages instead, and those are not
# carried over to the runtime stage — leaving polars/pyarrow/etc. missing.
RUN /usr/bin/python -m venv ${VIRTUAL_ENV} && \
    ${VIRTUAL_ENV}/bin/python -m pip install --no-cache-dir \
        "polars>=1.0.0,<2.0.0" \
        "pyarrow>=21.0.0,<22.0.0" \
        "datarobot>=3.10.0,<3.11.0" \
        "requests>=2.32.4,<3.0.0" \
        "httpx>=0.28.1,<1.0.0" && \
    find ${VIRTUAL_ENV} -type d -name '__pycache__' -exec rm -rf {} +

# Production (non-dev) chain-guard image for the final, minimal runtime.
FROM datarobot/mirror_chainguard_datarobot.com_python-fips:3.12

USER root

COPY --from=build /opt/venv /opt/venv
COPY --from=build /bin/sh /bin/sh
COPY --from=build /bin/mkdir /bin/mkdir
COPY --from=build /bin/chown /bin/chown

RUN /bin/mkdir -p /opt/dr_sandbox && /bin/chown 65534:65534 /opt/dr_sandbox
COPY runner.py /opt/dr_sandbox/runner.py

ENV VIRTUAL_ENV=/opt/venv
ENV PATH=${VIRTUAL_ENV}/bin:${PATH}
ENV PYTHONPATH=/opt/venv/lib/python3.12/site-packages:/usr/lib/python3.12/site-packages
ENV PYTHONUNBUFFERED=1

USER 65534:65534
WORKDIR /opt/dr_sandbox

# The chainguard base sets ENTRYPOINT ["/usr/bin/python"]; reset it to the venv
# interpreter (which has polars/pyarrow/datarobot) and pass only the script as
# CMD. Leaving the inherited entrypoint in place would run
# ``python /usr/bin/python /opt/dr_sandbox/runner.py`` and fail immediately.
ENTRYPOINT ["/opt/venv/bin/python"]
CMD ["/opt/dr_sandbox/runner.py"]
