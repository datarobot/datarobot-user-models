# This is a private chain-guard development image that is stored in DataRobot's private registry.
# Replace it with your own development chain-gaurd image if you build your own.
ARG PRODUCTION_BASE_IMAGE=datarobotdev/mirror_chainguard_datarobot.com_python-fips:3.11
ARG DEVELOPMENT_BASE_IMAGE=${PRODUCTION_BASE_IMAGE}-dev

FROM ${DEVELOPMENT_BASE_IMAGE} AS build

FROM ${PRODUCTION_BASE_IMAGE}

USER root

# Required to run the entrypoint script
COPY --from=build /bin/sh /bin/sh

# Required to change the ownership of copied files into the managed-image
COPY --from=build /bin/chown /bin/chown

# Required to change the permissions of the 'start_server.sh' that is copied into the managed-image
COPY --from=build /bin/chmod /bin/chmod

# Required for DR backend to create /opt/code (Maybe required by applications other than custom-models)
COPY --from=build /bin/mkdir /bin/mkdir

# Required for custom-models to install dependencies
COPY --from=build /usr/bin/pip /usr/bin/pip

# Just for convenience
COPY --from=build /bin/ls /bin/ls

COPY requirements.txt requirements.txt

ENV VIRTUAL_ENV=/opt/venv

RUN sh -c "python -m venv ${VIRTUAL_ENV} && \
    . ${VIRTUAL_ENV}/bin/activate && \
    python -m ensurepip --default-pip && \
    python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt"

ENV PATH=${VIRTUAL_ENV}/bin:${PATH}
ENV HOME=/opt
ENV CODE_DIR=/opt/code
ENV ADDRESS=0.0.0.0:8080

# MARK: FUNCTIONAL-TEST-ADD-HERE. (This line is used by DRUM functional test automation and can be safely ignored.)

# This makes print statements show up in the logs API
ENV WITH_ERROR_SERVER=1 \
    PYTHONUNBUFFERED=1


COPY ./*.sh ${CODE_DIR}/
WORKDIR ${CODE_DIR}

ENTRYPOINT ["sh", "-c", "${CODE_DIR}/start_server.sh \"$@\"", "--"]