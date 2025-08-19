# This is a private chain-guard development image that is stored in DataRobot's private registry.
# Replace it with your own development chain-gaurd image if you build your own.
FROM datarobotdev/mirror_chainguard_datarobot.com_python-fips:3.11-dev AS build
ENV VIRTUAL_ENV=/opt/venv

USER root
RUN python -m venv ${VIRTUAL_ENV} && \
    ${VIRTUAL_ENV}/bin/python -m ensurepip --default-pip --upgrade

COPY requirements.txt requirements.txt
RUN ${VIRTUAL_ENV}/bin/python -m pip install -r requirements.txt


# This is a private production chain-guard image that is stored in DataRobot's private registry.
# Replace it with your own production chain-gaurd image if you build your own.
FROM datarobotdev/mirror_chainguard_datarobot.com_python-fips:3.11
ENV VIRTUAL_ENV=/opt/venv

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

# Required for DR backend to create /opt/code (Maybe required by applications other than custom-models)
COPY --from=build /bin/mkdir /bin/mkdir

# Just for convenience
COPY --from=build /bin/ls /bin/ls

COPY --from=build ${VIRTUAL_ENV} ${VIRTUAL_ENV}
ENV PATH=${VIRTUAL_ENV}/bin:${PATH}
ENV OPENAI_HOST=localhost
ENV OPENAI_PORT=8000
ENV CODE_DIR=/opt/code
ENV ADDRESS=0.0.0.0:8080
ENV WITH_ERROR_SERVER=1

RUN mkdir -p ${CODE_DIR} \
    && chmod a+rwX ${CODE_DIR}

# This makes print statements show up in the logs API
ENV PYTHONUNBUFFERED=1

WORKDIR ${CODE_DIR}
COPY . ${CODE_DIR}/

# We have found performance to be best with 10 workers and by setting it this way, users can
# still override it with the CUSTOM_MODEL_WORKERS runtime param.
ENV MAX_WORKERS=10
ENTRYPOINT ["/opt/code/start_server.sh"]
