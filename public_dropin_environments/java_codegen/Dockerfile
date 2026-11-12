ARG PYTHON_VERSION=3.13

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
    find ${VIRTUAL_ENV} -type d -name '__pycache__' -exec rm -rf {} +"

# The JDK (not just JRE) is required because Py4J calls a Java method (o0.configure) that tries to execute 'javac'.
# Pin OpenJDK 11.0.30+ (Oracle CPU Jan 2026 / OpenJDK 11u) — addresses CVE-2026-21932, CVE-2026-21945 on the 11 train.
# Wolfi epoch for 11.0.30 GA is typically r6; override at build if your mirror uses a different release suffix.
# RAPTOR-19452: 11.0.32-r0 -> 11.0.32.1-r0 (JAVA_VERSION 11.0.32.1, 2026-08-18).
ARG OPENJDK11_APK_VERSION=11.0.32.1-r6
# java-cacerts is pinned so a lagging mirror can't ship the compromised eMudhra roots (PRDSEC-2816).
RUN apk add --no-cache openjdk-11=${OPENJDK11_APK_VERSION} 'java-cacerts>=20260909-r3'

# FIPS 140-3 (RAPTOR-21063): the JVM gets the Bouncy Castle FIPS provider (FIPS 140-3 module,
# approved-only mode) in place of the JDK's own crypto providers, and a startup self-test agent.
# The jars come from Maven Central, pinned by SHA-256 (fips_jvm/fetch_bcfips.py); java.security is
# this JDK's own file with the provider list switched (fips_jvm/apply_bcfips_security.sh).
COPY fips_jvm/ /tmp/fips_jvm/
RUN /usr/bin/python /tmp/fips_jvm/fetch_bcfips.py /usr/share/java/bouncycastle-fips && \
    sh /tmp/fips_jvm/apply_bcfips_security.sh /usr/lib/jvm/java-11-openjdk/conf/security/java.security && \
    cd /tmp/fips_jvm && \
    /usr/lib/jvm/java-11-openjdk/bin/javac -cp '/usr/share/java/bouncycastle-fips/*' -d out \
      com/datarobot/fips/FipsSelfTest.java && \
    printf 'Premain-Class: com.datarobot.fips.FipsSelfTest\n' > manifest && \
    mkdir -p /usr/share/java/datarobot-fips && \
    /usr/lib/jvm/java-11-openjdk/bin/jar --create --file /usr/share/java/datarobot-fips/datarobot-fips-selftest.jar \
      --manifest manifest -C out com && \
    rm -rf /tmp/fips_jvm

# This is a private production chain-guard image that is stored in DataRobot's private registry.
# Replace it with your own production chain-guard image if you build your own.
# RAPTOR-19887: rebuild for CVE-2026-63073 - libcrypto3/libssl3 3.6.3-r5 -> 3.6.4-r1.
# Both ship from this rolling base and are not pinned here, so bumping
# environmentVersionId in env_info.json is the whole fix.
FROM datarobot/mirror_chainguard_datarobot.com_python-fips:${PYTHON_VERSION}

USER root

# Copy the JDK from the development stage (its java.security switched to Bouncy Castle FIPS), the
# Bouncy Castle FIPS jars and the JVM FIPS self-test agent (RAPTOR-21063).
COPY --from=build /usr/lib/jvm/java-11-openjdk /usr/lib/jvm/java-11-openjdk
COPY --from=build /usr/share/java/bouncycastle-fips /usr/share/java/bouncycastle-fips
COPY --from=build /usr/share/java/datarobot-fips /usr/share/java/datarobot-fips

# The JDK's lib/security/cacerts is a symlink to this file from java-cacerts (pulled in by
# openjdk-11 in the build stage). Without it Java has no default trust store, so every TLS
# connection fails. java-cacerts 20260909-r3 or later drops the compromised eMudhra roots
# (PRDSEC-2816).
COPY --from=build /etc/ssl/certs/java/cacerts /etc/ssl/certs/java/cacerts

# Most of the binaries below are just symlinks to busybox and some OCI build tools follow
# symlinks (Docker buildkit) and some do not (Kaniko) so copy this in to be safe.
COPY --from=build /usr/bin/busybox /usr/bin/busybox

# Required to run the entrypoint script
COPY --from=build /bin/sh /bin/sh
# Not present in non-dev image now, so /bin/sh call fails without this
COPY --from=build /usr/lib/libcrypt.so.1 /usr/lib/libcrypt.so.1

# Fail the build if Java's default trust store doesn't resolve (functional tests use plain HTTP
# and would not notice a dangling cacerts symlink).
RUN /usr/lib/jvm/java-11-openjdk/bin/keytool -list -cacerts -storepass changeit >/dev/null

# Required to change the ownership of copied files into the managed-image
COPY --from=build /bin/chown /bin/chown

# Required to change the ownership for Custom Models PPS
COPY --from=build /bin/chgrp /bin/chgrp

# Required to change the permissions of the 'start_server.sh' that is copied into the managed-image
COPY --from=build /bin/chmod /bin/chmod

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


ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk
ENV PATH=${VIRTUAL_ENV}/bin:${JAVA_HOME}/bin:${PATH}

# FIPS 140-3 (RAPTOR-21063): every JVM (the DRUM Py4J gateway, javac) loads Bouncy Castle FIPS from
# the module path and runs the startup self-test agent first. With DATAROBOT_FIPS_ENFORCE=true the
# agent halts the JVM with exit 78 and a "[datarobot-fips] FATAL" line unless the FIPS module passed
# its power-on self-tests and is the first provider, in approved-only mode. The add-exports are the
# ones Chainguard's jdk-fips image sets for Bouncy Castle FIPS on JDK 11.
ENV JAVA_TOOL_OPTIONS="--module-path=/usr/share/java/bouncycastle-fips -javaagent:/usr/share/java/datarobot-fips/datarobot-fips-selftest.jar"
ENV JDK_JAVA_OPTIONS="--add-exports=java.base/sun.security.internal.spec=ALL-UNNAMED --add-exports=java.base/sun.security.provider=ALL-UNNAMED"
# Fail the build unless the self-test passes here, and unless it refuses a JVM with a non-FIPS
# provider (SunJCE) added, so an agent that passes everything can't ship.
RUN DATAROBOT_FIPS_ENFORCE=true java -version 2>&1 | busybox grep -q -e '^\[datarobot-fips\] JVM FIPS self-test passed' && \
    printf 'security.provider.9=SunJCE\n' > /tmp/sunjce.security && \
    { DATAROBOT_FIPS_ENFORCE=true java -Djava.security.properties=/tmp/sunjce.security -version >/dev/null 2>&1; \
      test $? -eq 78; } && \
    rm -f /tmp/sunjce.security
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
ARG PYTHON_VERSION
ENV PYTHONPATH=/etc/datarobot-fips:/opt/venv/lib/python${PYTHON_VERSION}/site-packages:/usr/lib/python${PYTHON_VERSION}/site-packages

# Gunicorn configuration
ENV MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE='{"type": "string", "payload": "gunicorn"}'
ENV MLOPS_RUNTIME_PARAM_CUSTOM_MODEL_WORKERS='{"type": "numeric", "payload": 1}'

COPY ./*.sh ${CODE_DIR}/
WORKDIR ${CODE_DIR}

ENTRYPOINT ["sh", "-c", "exec ${CODE_DIR}/start_server.sh \"$@\"", "--"]
