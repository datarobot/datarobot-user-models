# This is a private chain-guard development image that is stored in DataRobot's private registry.
# Replace it with your own development chain-gaurd image if you build your own.
ARG BASE_ROOT_IMAGE=datarobotdev/mirror_chainguard_datarobot.com_python-fips:3.11-dev
FROM ${BASE_ROOT_IMAGE} AS build

USER root

# Install R and required R packages using apk and Rscript
RUN apk add --no-cache \
    R \
    R-dev \
    openblas \
    harfbuzz \
    pkgconf \
    fontconfig fontconfig-dev \
    libsodium-dev \
    openssl-dev \
    libxml2-dev \
    freetype-dev \
    harfbuzz-dev \
    fribidi-dev \
    libpng-dev \
    tiff-dev \
    libjpeg-turbo-dev && \
    ln -s /usr/lib/R/lib /usr/lib/R/lib64

ENV R_HOME=/usr/lib/R
ENV LD_LIBRARY_PATH=${R_HOME}/lib:/usr/lib

RUN Rscript -e "install.packages( \
        c('caret', 'recipes', 'glmnet', 'Rook', 'rjson', 'e1071', 'tidyverse', 'devtools'), \
        repos='https://cloud.r-project.org', \
        Ncpus=4 \
    )" && \
    Rscript -e "library(caret); install.packages( \
        unique(modelLookup()[modelLookup()[['forReg']] | modelLookup()[['forClass']], 'model']), \
        repos='https://cloud.r-project.org', \
        Ncpus=4 \
    )"

ENV VIRTUAL_ENV=/opt/venv

RUN python -m venv ${VIRTUAL_ENV} && \
    ${VIRTUAL_ENV}/bin/python -m ensurepip --default-pip --upgrade

COPY requirements.txt requirements.txt

RUN ${VIRTUAL_ENV}/bin/python -m pip install --no-cache-dir -r requirements.txt && \
    find ${VIRTUAL_ENV} -type d -name '__pycache__' -exec rm -rf {} +


# This is a private production chain-guard image that is stored in DataRobot's private registry.
# Replace it with your own production chain-gaurd image if you build your own.
FROM datarobotdev/mirror_chainguard_datarobot.com_python-fips:3.11

USER root

# Copy R libraries and binaries from the build stage
COPY --from=build /usr/bin/R /usr/bin/R
COPY --from=build /usr/bin/Rscript /usr/bin/Rscript
COPY --from=build /usr/lib/R /usr/lib/R
COPY --from=build /usr/lib/libopenblas* /usr/lib/
COPY --from=build /usr/lib/libgfortran* /usr/lib/
COPY --from=build /usr/lib/libquadmath* /usr/lib/
COPY --from=build /usr/lib/libpcre2-8* /usr/lib/
COPY --from=build /usr/lib/libiconv* /usr/lib/
COPY --from=build /usr/lib/libicuuc* /usr/lib/
COPY --from=build /usr/lib/libicui18n* /usr/lib/
COPY --from=build /usr/lib/libgomp* /usr/lib/
COPY --from=build /usr/lib/libicudata* /usr/lib/
COPY --from=build /usr/lib/libicu* /usr/lib/

# 'curl' is required by R to download packages
COPY --from=build /usr/lib/libcurl* /usr/lib/
COPY --from=build /usr/lib/libtiff* /usr/lib/
COPY --from=build /usr/lib/libjpeg* /usr/lib/
COPY --from=build /usr/lib/libnghttp2* /usr/lib/
COPY --from=build /usr/lib/libpsl* /usr/lib/
COPY --from=build /usr/lib/libgssapi_krb5* /usr/lib/
COPY --from=build /usr/lib/libldap* /usr/lib/
COPY --from=build /usr/lib/liblber* /usr/lib/
COPY --from=build /usr/lib/libbrotlidec* /usr/lib/
COPY --from=build /usr/lib/libunistring* /usr/lib/
COPY --from=build /usr/lib/libidn2* /usr/lib/
COPY --from=build /usr/lib/libkrb5* /usr/lib/
COPY --from=build /usr/lib/libk5crypto* /usr/lib/
COPY --from=build /usr/lib/libcom_err* /usr/lib/
COPY --from=build /usr/lib/libkeyutils* /usr/lib/
COPY --from=build /usr/lib/libevent-2.1* /usr/lib/
COPY --from=build /usr/lib/libsasl2* /usr/lib/
COPY --from=build /usr/lib/libbrotlicommon* /usr/lib/
COPY --from=build /usr/lib/libzstd* /usr/lib/

COPY --from=build /etc/R /etc/R
COPY --from=build /usr/share/R /usr/share/R
COPY --from=build /usr/share/icu /usr/share/icu

# Required by R
COPY --from=build /bin/sed /bin/sed
COPY --from=build /bin/rm /bin/rm
COPY --from=build /bin/echo /bin/echo
COPY --from=build /bin/mv /bin/mv
COPY --from=build /usr/bin/which /usr/bin/which
COPY --from=build /usr/bin/make /usr/bin/make
COPY --from=build /usr/bin/curl /usr/bin/curl

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
ENV R_HOME=/usr/lib/R

COPY ensure_sh_shebang.py ensure_sh_shebang.py
# Copy preinstalled virtualenv
COPY --from=build --chown=nonroot ${VIRTUAL_ENV} ${VIRTUAL_ENV}

RUN python ensure_sh_shebang.py ${R_HOME}/bin
RUN mkdir -p /usr/share/doc/R/html

ENV PATH=${VIRTUAL_ENV}/bin:${PATH}
ENV LD_LIBRARY_PATH=${R_HOME}/lib:/usr/lib
ENV HOME=/opt
ENV CODE_DIR=/opt/code
ENV ADDRESS=0.0.0.0:8080

# This makes print statements show up in the logs API
ENV WITH_ERROR_SERVER=1 \
    PYTHONUNBUFFERED=1

COPY ./*.sh ${CODE_DIR}/
WORKDIR ${CODE_DIR}

USER nonroot

ENTRYPOINT ["sh", "-c", "exec ${CODE_DIR}/start_server.sh \"$@\"", "--"]
