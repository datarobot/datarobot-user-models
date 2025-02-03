# This is a private chain-guard development image that is stored in DataRobot's private registry.
# Replace it with your own development chain-gaurd image if you build your own.
ARG BASE_ROOT_IMAGE=datarobotdev/mirror_chainguard_datarobot.com_python-fips:3.11-dev
FROM ${BASE_ROOT_IMAGE} AS build

USER root

# Fix these CVEs: CVE-2025-0395, CVE-2024-13176, CVE-2025-0395
RUN apk update && apk upgrade --available && \
    apk add --no-cache glibc=2.40-r6 glibc-dev=2.40-r6 glibc-locale-posix=2.40-r6 \
    ld-linux=2.40-r6 libcrypt1=2.40-r6 nss-db=2.40-r6 nss-hesiod=2.40-r6 \
    openssl=3.4.0-r6 libcrypto3=3.4.0-r6 libssl3=3.4.0-r6 && \
    rm -rf /var/cache/apk/* && \
    mkdir -p /opt/code  # Required for custom-models

USER nonroot

ENV HOME=/home/nonroot
WORKDIR ${HOME}

COPY requirements.txt requirements.txt
RUN pip install -U pip && \
    pip install --no-cache-dir -r requirements.txt  && \
    rm -rf requirements.txt

# MARK: FUNCTIONAL-TEST-ADD-HERE. (This line is used by DRUM functional test automation and can be safely ignored.)

# This is a private production chain-guard image that is stored in DataRobot's private registry.
# Replace it with your own production chain-gaurd image if you build your own.
FROM datarobotdev/mirror_chainguard_datarobot.com_python-fips:3.11

USER root

# Copy updated libraries free of CVEs
COPY --from=build /lib /lib
COPY --from=build /usr/lib /usr/lib

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

# The '/opt/code' is being copied by the root, although the files are owned by nonroot
COPY --from=build --chown=nonroot:nonroot /opt/code /opt/code

USER nonroot

ENV HOME=/home/nonroot
ENV PATH=${HOME}/.local/bin:${PATH}
ENV CODE_DIR=/opt/code
ENV ADDRESS=0.0.0.0:8080

# This makes print statements show up in the logs API
ENV WITH_ERROR_SERVER=1 \
    PYTHONUNBUFFERED=1

COPY --from=build ${HOME} ${HOME}
COPY --chown=nonroot:nonroot ./*.sh ${CODE_DIR}/
WORKDIR ${CODE_DIR}

ENTRYPOINT ["sh", "-c", "${CODE_DIR}/start_server.sh \"$@\"", "--"]
