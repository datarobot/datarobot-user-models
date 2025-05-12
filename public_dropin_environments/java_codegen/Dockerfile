# This is a private chain-guard development image that is stored in DataRobot's private registry.
# Replace it with your own development chain-gaurd image if you build your own.
ARG BASE_ROOT_IMAGE=datarobotdev/mirror_chainguard_datarobot.com_python-fips:3.11-dev
FROM ${BASE_ROOT_IMAGE} AS build

USER root

RUN apk add --no-cache openjdk-11

ENV VIRTUAL_ENV=/opt/venv

RUN python -m venv ${VIRTUAL_ENV} && \
    ${VIRTUAL_ENV}/bin/python -m ensurepip --default-pip --upgrade

COPY requirements.txt requirements.txt
RUN ${VIRTUAL_ENV}/bin/python -m pip install --no-cache-dir -r requirements.txt && \
    find ${VIRTUAL_ENV} -type d -name '__pycache__' -exec rm -rf {} +

# The JDK (not just JRE) is required because Py4J calls a Java method (o0.configure) that tries to execute 'javac'.

# This is a private production chain-guard image that is stored in DataRobot's private registry.
# Replace it with your own production chain-gaurd image if you build your own.
FROM datarobotdev/mirror_chainguard_datarobot.com_python-fips:3.11

# Copy the JRE from the development stage
COPY --from=build /usr/lib/jvm/java-11-openjdk /usr/lib/jvm/java-11-openjdk

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

ENV VIRTUAL_ENV=/opt/venv
# Copy preinstalled virtualenv
COPY --from=build --chown=nonroot ${VIRTUAL_ENV} ${VIRTUAL_ENV}

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk
ENV PATH=${VIRTUAL_ENV}/bin:${JAVA_HOME}/bin:${PATH}
ENV HOME=/opt
ENV CODE_DIR=/opt/code
ENV ADDRESS=0.0.0.0:8080

# MARK: FUNCTIONAL-TEST-ADD-HERE. (This line is used by DRUM functional test automation and can be safely ignored.)

# This makes print statements show up in the logs API
ENV WITH_ERROR_SERVER=1 \
    PYTHONUNBUFFERED=1


COPY ./*.sh ${CODE_DIR}/
WORKDIR ${CODE_DIR}

USER nonroot

ENTRYPOINT ["sh", "-c", "exec ${CODE_DIR}/start_server.sh \"$@\"", "--"]
