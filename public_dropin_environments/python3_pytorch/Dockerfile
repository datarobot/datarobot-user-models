# This is a private chain-guard development image that is stored in DataRobot's private registry.
# Replace it with your own development chain-gaurd image if you build your own.
ARG BASE_ROOT_IMAGE=datarobot/mirror_chainguard_datarobot.com_python-fips:3.11-dev
FROM ${BASE_ROOT_IMAGE} AS build

# This is a private production chain-guard image that is stored in DataRobot's private registry.
# Replace it with your own production chain-gaurd image if you build your own.
FROM datarobot/mirror_chainguard_datarobot.com_python-fips:3.11

USER root

# Most of the binaries below are just symlinks to busybox and some OCI build tools follow
# symlinks (Docker buildkit) and some do not (Kaniko) so copy this in to be safe.
COPY --from=build /usr/bin/busybox /usr/bin/busybox

# Required to run the entrypoint script
COPY --from=build /bin/sh /bin/sh

# Required to change the ownership of copied files into the managed-image
COPY --from=build /bin/chown /bin/chown

# Required to change the ownership for Custom Models PPS
COPY --from=build /bin/chgrp /bin/chgrp

# Required to change the permissions of the 'start_server.sh' that is copied into the managed-image
COPY --from=build /bin/chmod /bin/chmod

# Required for DR backend to create /opt/code (Maybe required by applications other than custom-models)
COPY --from=build /bin/mkdir /bin/mkdir

# Required for custom-models to install dependencies
COPY --from=build /usr/bin/pip /usr/bin/pip
COPY --from=build /usr/lib/python3.11/site-packages/pip /usr/lib/python3.11/site-packages/pip

# Cleanup '__pycache__' directories. It solves an AsymmetricPrivateKey scanning error.
COPY --from=build /usr/bin/rm /usr/bin/rm
COPY --from=build /usr/bin/find /usr/bin/find

# Just for convenience
COPY --from=build /bin/ls /bin/ls

COPY requirements.txt requirements.txt

ENV VIRTUAL_ENV=/opt/venv

RUN sh -c "/usr/bin/python -m venv --without-pip ${VIRTUAL_ENV} && \
    . ${VIRTUAL_ENV}/bin/activate && \
    /usr/bin/python -m pip install --no-cache-dir -r requirements.txt && \
    find ${VIRTUAL_ENV} -type d -name '__pycache__' -exec rm -rf {} +"

ENV PATH=${VIRTUAL_ENV}/bin:${PATH}
ENV HOME=/opt
ENV CODE_DIR=/opt/code
ENV ADDRESS=0.0.0.0:8080

# MARK: FUNCTIONAL-TEST-ADD-HERE. (This line is used by DRUM functional test automation and can be safely ignored.)

# This makes print statements show up in the logs API
ENV WITH_ERROR_SERVER=0 \
    PYTHONUNBUFFERED=1


COPY ./*.sh ${CODE_DIR}/
WORKDIR ${CODE_DIR}

ENTRYPOINT ["sh", "-c", "exec ${CODE_DIR}/start_server.sh \"$@\"", "--"]
