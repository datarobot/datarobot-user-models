ARG PYTHON_VERSION=3.12

# This is a private chain-guard development image that is stored in DataRobot's private registry.
# Replace it with your own development chain-guard image if you build your own.
FROM datarobot/mirror_chainguard_datarobot.com_python-fips:${PYTHON_VERSION}-dev AS build

USER root
RUN apk update && apk add rust openssl-dev

# Pillow is built from source (--no-binary pillow in requirements.txt): the PyPI wheel bundles
# libbrotlidec 1.2.0 in pillow.libs (CVE-2026-111497). Built here it links the image's
# FreeType and codec libraries, and that FreeType links the system libbrotlidec. Of these the
# production image ships only Brotli and bzip2 (the final-stage check requires Brotli 1.2.1 or
# later), so the rest are staged in /opt/pillow-libs below.
RUN apk add --no-cache build-base zlib-dev libjpeg-turbo-dev libpng-dev libwebp-dev tiff-dev \
    openjpeg-dev lcms2-dev freetype-dev brotli-dev libavif-dev

COPY requirements.txt requirements.txt

ENV VIRTUAL_ENV=/opt/venv

RUN sh -c "/usr/bin/python -m venv ${VIRTUAL_ENV} && \
    . ${VIRTUAL_ENV}/bin/activate && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    find ${VIRTUAL_ENV} -path '*/site-packages/datarobot_drum/*.jar' -delete && \
    find ${VIRTUAL_ENV} -type d -name '__pycache__' -exec rm -rf {} +"

# Stage the shared libraries the source-built Pillow needs at runtime, with each one's apk database
# record and SBOM so image scanners still see the package it came from (the -dev packages pull in
# libxcb, so Pillow builds its XCB module too). cp -L ships each soname as one regular file; the apk
# record lists the package's every file, so apk audit reports the rest missing, while scanners,
# which key on package name and version, see the right package.
ARG PILLOW_RUNTIME_LIBS="libfreetype.so.6 libjpeg.so.8 liblcms2.so.2 libopenjp2.so.7 libpng16.so.16 libtiff.so.6 libwebp.so.7 libwebpdemux.so.2 libwebpmux.so.3 libsharpyuv.so.0 libzstd.so.1 libavif.so.16 libdav1d.so.7 libaom.so.3 libxcb.so.1 libXau.so.6 libXdmcp.so.6"
RUN ${VIRTUAL_ENV}/bin/python -Bc "import ctypes, os, PIL; from PIL import features; missing = [f for f in ('freetype2', 'littlecms2', 'webp', 'jpg', 'jpg_2000', 'zlib', 'libtiff', 'avif', 'xcb') if not features.check(f)]; bundled = os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(PIL.__file__)), 'pillow.libs')); brotli = ctypes.CDLL('libbrotlidec.so.1').BrotliDecoderVersion(); assert not missing and not bundled and brotli >= 0x01002001, f'Pillow source build is wrong: missing={missing} bundled_libs={bundled} brotli={brotli >> 24}.{(brotli >> 12) & 0xFFF}.{brotli & 0xFFF}'" && \
    mkdir -p /opt/pillow-libs/lib /opt/pillow-libs/sbom && cd /usr/lib && \
    for lib in ${PILLOW_RUNTIME_LIBS}; do cp -L "${lib}" /opt/pillow-libs/lib/ || exit 1; done && \
    owners="$(for lib in ${PILLOW_RUNTIME_LIBS}; do apk info -W "${lib}"; done)" && \
    test "$(printf '%s\n' "${owners}" | grep -c ' is owned by ')" -eq "$(echo ${PILLOW_RUNTIME_LIBS} | wc -w)" && \
    for pkg in $(printf '%s\n' "${owners}" | sed 's/.* is owned by //' | sort -u); do \
        cp "/var/lib/db/sbom/${pkg}.spdx.json" /opt/pillow-libs/sbom/ && \
        awk -v pkg="${pkg}" 'BEGIN{RS="";ORS="\n\n"} {p="";v="";n=split($0,L,"\n");for(i=1;i<=n;i++){if(L[i]~/^P:/)p=substr(L[i],3);if(L[i]~/^V:/)v=substr(L[i],3)} if(p"-"v==pkg)print}' /lib/apk/db/installed >> /opt/pillow-libs/apk-installed || exit 1; \
    done && \
    test "$(grep -c '^P:' /opt/pillow-libs/apk-installed)" -eq "$(ls /opt/pillow-libs/sbom | wc -l)"

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

# Shared libraries for the source-built Pillow, with their apk records and SBOMs (see the build stage).
COPY --from=build /opt/pillow-libs/lib/ /usr/lib/
COPY --from=build /opt/pillow-libs/sbom/ /var/lib/db/sbom/
COPY --from=build /opt/pillow-libs/apk-installed /opt/pillow-libs/apk-installed
RUN /opt/venv/bin/python -Bc "import re, shutil; staged = open('/opt/pillow-libs/apk-installed').read(); db = open('/lib/apk/db/installed').read(); names = set(re.findall(r'^P:(.*)$', staged, re.M)); present = sorted(names & set(re.findall(r'^P:(.*)$', db, re.M))); assert not present, f'staged packages the base image already ships: {present}'; pad = '' if db.endswith('\n\n') else ('\n' if db.endswith('\n') else '\n\n'); open('/lib/apk/db/installed', 'a').write(pad + staged); shutil.rmtree('/opt/pillow-libs')"
RUN /opt/venv/bin/python -Bc "import ctypes, os, PIL; from PIL import features; missing = [f for f in ('freetype2', 'littlecms2', 'webp', 'jpg', 'jpg_2000', 'zlib', 'libtiff', 'avif', 'xcb') if not features.check(f)]; bundled = os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(PIL.__file__)), 'pillow.libs')); brotli = ctypes.CDLL('libbrotlidec.so.1').BrotliDecoderVersion(); assert not missing and not bundled and brotli >= 0x01002001, f'Pillow source build is wrong: missing={missing} bundled_libs={bundled} brotli={brotli >> 24}.{(brotli >> 12) & 0xFFF}.{brotli & 0xFFF}'"
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

# Gunicorn configuration
ENV MLOPS_RUNTIME_PARAM_DRUM_SERVER_TYPE='{"type": "string", "payload": "gunicorn"}'
ENV MLOPS_RUNTIME_PARAM_CUSTOM_MODEL_WORKERS='{"type": "numeric", "payload": 1}'

COPY ./*.sh ${CODE_DIR}/
WORKDIR ${CODE_DIR}

ENTRYPOINT ["sh", "-c", "exec ${CODE_DIR}/start_server.sh \"$@\"", "--"]
