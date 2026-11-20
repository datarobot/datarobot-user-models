ARG PYTHON_VERSION=3.12

# This is a private chain-guard development image that is stored in DataRobot's private registry.
# Replace it with your own development chain-guard image if you build your own.
FROM datarobot/mirror_chainguard_datarobot.com_python-fips:${PYTHON_VERSION}-dev AS build

USER root
RUN apk update && apk add rust openssl-dev

# Pillow is built from source (--no-binary pillow in requirements.txt): the PyPI wheel bundles
# libbrotlidec 1.2.0 in pillow.libs (CVE-2026-111497). Built here it links the image's
# FreeType and codec libraries, and that FreeType links the system libbrotlidec 1.2.1. The
# production image ships none of these libraries, so they are staged in /opt/pillow-libs below.
RUN apk add --no-cache build-base zlib-dev libjpeg-turbo-dev libpng-dev libwebp-dev tiff-dev \
    openjpeg-dev lcms2-dev freetype-dev brotli-dev

COPY requirements.txt requirements.txt

ENV VIRTUAL_ENV=/opt/venv

RUN sh -c "/usr/bin/python -m venv ${VIRTUAL_ENV} && \
    . ${VIRTUAL_ENV}/bin/activate && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    find ${VIRTUAL_ENV} -path '*/site-packages/datarobot_drum/*.jar' -delete && \
    find ${VIRTUAL_ENV} -type d -name '__pycache__' -exec rm -rf {} +"

# Stage the shared libraries the source-built Pillow needs at runtime (the -dev packages pull in
# libxcb, so Pillow builds its XCB module too).
RUN ${VIRTUAL_ENV}/bin/python -Bc "import os, PIL; from PIL import features; missing = [f for f in ('freetype2', 'littlecms2', 'webp', 'jpg', 'jpg_2000', 'zlib', 'libtiff') if not features.check(f)]; bundled = os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(PIL.__file__)), 'pillow.libs')); assert not missing and not bundled, f'Pillow source build is wrong: missing={missing} bundled_libs={bundled}'" && \
    mkdir -p /opt/pillow-libs && cd /usr/lib && \
    cp -L libfreetype.so.6 libjpeg.so.8 liblcms2.so.2 libopenjp2.so.7 libpng16.so.16 libtiff.so.6 libwebp.so.7 libwebpdemux.so.2 libwebpmux.so.3 libsharpyuv.so.0 libzstd.so.1 \
          libxcb.so.1 libXau.so.6 libXdmcp.so.6 /opt/pillow-libs/

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

# Shared libraries for the source-built Pillow (see the build stage).
COPY --from=build /opt/pillow-libs/ /usr/lib/
RUN /opt/venv/bin/python -Bc "import os, PIL; from PIL import features; missing = [f for f in ('freetype2', 'littlecms2', 'webp', 'jpg', 'jpg_2000', 'zlib', 'libtiff') if not features.check(f)]; bundled = os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(PIL.__file__)), 'pillow.libs')); assert not missing and not bundled, f'Pillow source build is wrong: missing={missing} bundled_libs={bundled}'"
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
ARG PYTHON_VERSION
ENV PYTHONPATH=/etc/datarobot-fips:/opt/venv/lib/python${PYTHON_VERSION}/site-packages:/usr/lib/python${PYTHON_VERSION}/site-packages

# OMP_NUM_THREADS is an environment variable that controls the maximum number of threads used by programs that rely on OpenMP for parallel processing.
# In libraries like XGBoost, it limits the number of CPU threads used for computation, helping to prevent resource contention and hangs in multi-worker environments.
# Setting OMP_NUM_THREADS=1 forces XGBoost to use only one thread per process.
# If we need to use wekzeug max_workers > 1 in DRUM, we must set OMP_NUM_THREADS=1 to avoid potential hangs.
# Maybe the same would apply to gunicorn, when we test.
# ENV OMP_NUM_THREADS=1

# Gunicorn configuration
ENV MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE='{"type": "string", "payload": "gunicorn"}'
ENV MLOPS_RUNTIME_PARAM_CUSTOM_MODEL_WORKERS='{"type": "numeric", "payload": 1}'

COPY ./*.sh ${CODE_DIR}/
WORKDIR ${CODE_DIR}

ENTRYPOINT ["sh", "-c", "exec ${CODE_DIR}/start_server.sh \"$@\"", "--"]
