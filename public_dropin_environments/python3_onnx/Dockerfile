ARG PYTHON_VERSION=3.12

# This is a private chain-guard development image that is stored in DataRobot's private registry.
# Replace it with your own development chain-guard image if you build your own.
FROM datarobot/mirror_chainguard_datarobot.com_python-fips:${PYTHON_VERSION}-dev AS build

USER root
RUN apk update && apk add rust openssl-dev

COPY requirements.txt requirements.txt

ENV VIRTUAL_ENV=/opt/venv

RUN sh -c "/usr/bin/python -m venv ${VIRTUAL_ENV} && \
    . ${VIRTUAL_ENV}/bin/activate && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    find ${VIRTUAL_ENV} -path '*/site-packages/datarobot_drum/*.jar' -delete && \
    find ${VIRTUAL_ENV} -type d -name '__pycache__' -exec rm -rf {} +"

# This is a private production chain-guard image that is stored in DataRobot's private registry.
# Replace it with your own production chain-guard image if you build your own.
FROM datarobot/mirror_chainguard_datarobot.com_python-fips:${PYTHON_VERSION}

USER root

# Most of the binaries below are just symlinks to busybox and some OCI build tools follow
# symlinks (Docker buildkit) and some do not (Kaniko) so copy this in to be safe.
COPY --from=build /usr/bin/busybox /usr/bin/busybox

# Required to run the entrypoint script
COPY --from=build /bin/sh /bin/sh

# Required to change the ownership of copied files into the managed-image
COPY --from=build /bin/chown /bin/chown

# Required to change the permissions of the 'start_server.sh' that is copied into the managed-image
COPY --from=build /bin/chmod /bin/chmod

# Required to change the ownership for Custom Models PPS
COPY --from=build /bin/chgrp /bin/chgrp

# Required for DR backend to create /opt/code (Maybe required by applications other than custom-models)
COPY --from=build /bin/mkdir /bin/mkdir

# Required for custom-models to install dependencies
ARG PYTHON_VERSION
COPY --from=build /usr/lib/python${PYTHON_VERSION}/site-packages/pip /usr/lib/python${PYTHON_VERSION}/site-packages/pip

# Cleanup '__pycache__' directories. It solves an AsymmetricPrivateKey scanning error.
COPY --from=build /usr/bin/rm /usr/bin/rm
COPY --from=build /usr/bin/find /usr/bin/find

# Just for convenience
COPY --from=build /bin/ls /bin/ls

# Required to source activate script in final image
COPY --from=build /usr/bin/uname /usr/bin/uname
# Copy the virtualenv built from the dev base image
COPY --from=build /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv

# FIPS Validation
# Validate openssl fips in the final image
COPY --from=build /usr/bin/openssl-fips-test /usr/bin/openssl-fips-test
RUN /usr/bin/openssl-fips-test
# Python cryptography fips validation
RUN sh -c 'printf "Validating python cryptography fips support\n"; \
  source /opt/venv/bin/activate; \
  python -Bc "from cryptography.hazmat.backends.openssl.backend import backend; assert backend._fips_enabled"'

ENV PATH=${VIRTUAL_ENV}/bin:${PATH}
ENV HOME=/opt
ENV CODE_DIR=/opt/code
ENV ADDRESS=0.0.0.0:8080

# MARK: FUNCTIONAL-TEST-ADD-HERE. (This line is used by DRUM functional test automation and can be safely ignored.)

# This makes print statements show up in the logs API
ENV WITH_ERROR_SERVER=0 \
    PYTHONUNBUFFERED=1

# FIPS 140-3 (INF-711): halt any Python process when the crypto module fails, at startup and
# periodically; with DATAROBOT_FIPS_ENFORCE=true also when OpenSSL is not in FIPS mode. Same files
# as the other environments with fips/; see fips/sitecustomize.py.
COPY ./fips /etc/datarobot-fips/

# Both system python and venv python should look at both venv and system site
# packages.
ENV PYTHONPATH=/etc/datarobot-fips:/opt/venv/lib/python${PYTHON_VERSION}/site-packages:/usr/lib/python${PYTHON_VERSION}/site-packages

# Gunicorn configuration
ENV MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE='{"type": "string", "payload": "gunicorn"}'
ENV MLOPS_RUNTIME_PARAM_CUSTOM_MODEL_WORKERS='{"type": "numeric", "payload": 1}'

COPY ./*.sh ${CODE_DIR}/
WORKDIR ${CODE_DIR}

ENTRYPOINT ["sh", "-c", "exec ${CODE_DIR}/start_server.sh \"$@\"", "--"]
