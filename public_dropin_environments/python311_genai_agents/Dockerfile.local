# Copyright 2025 DataRobot, Inc. and its affiliates.
# All rights reserved.
# DataRobot, Inc. Confidential.
# This is unpublished proprietary source code of DataRobot, Inc.
# and its affiliates.
# The copyright notice above does not evidence any actual or intended
# publication of such source code.

ARG WORKDIR=/etc/system/kernel
ARG AGENTDIR=/etc/system/kernel/agent
ARG VENV_PATH=${WORKDIR}/.venv

ARG UNAME=notebooks
ARG UID=10101
ARG GID=10101

# This reproduces a open source variant of https://images.chainguard.dev/directory/image/python
# hadolint ignore=DL3007
FROM cgr.dev/chainguard/wolfi-base:latest AS chainguard_python_dev

ARG PYTHON_VERSION=3.11
SHELL ["/bin/sh", "-o", "pipefail", "-c"]
RUN echo "[Python $PYTHON_VERSION]"

ENV PATH="${PATH}:/sbin"
RUN apk update && apk upgrade
# Wolfi python image
# hadolint ignore=DL3018,DL3059
RUN apk add --no-cache ca-certificates-bundle gdbm glibc glibc-locale-posix \
ld-linux libbz2-1 libcrypto3 libexpat1 libffi libgcc libssl3 libstdc++ \
mpdecimal ncurses ncurses-terminfo readline sqlite-libs \
wolfi-baselayout xz zlib
# Wolfi python dev image
# hadolint ignore=DL3018,DL3059
RUN apk add --no-cache apk-tools  bash  binutils  build-base  busybox  cyrus-sasl  \
gcc  git  glibc-dev  gmp  heimdal-libs  isl  keyutils-libs  \
krb5-conf  krb5-libs  libatomic  libbrotlicommon1  libbrotlidec1  \
libcom_err  libcrypt1  libcurl-openssl4  libgo  libgomp  libidn2  \
libldap  libnghttp2-14  libpcre2-8-0  libpsl  libquadmath  libstdc++-dev  \
libunistring  libverto  libxcrypt  libxcrypt-dev  libzstd1  \
linux-headers  make  mpc  mpfr  nss-db  nss-hesiod  \
openssf-compiler-options  pkgconf  posix-cc-wrappers  wget
# Python
# hadolint ignore=DL3018,DL3059
RUN apk add --no-cache \
    python-$PYTHON_VERSION \
    python-$PYTHON_VERSION-base \
    py$PYTHON_VERSION-pip \
    py3-pip-wheel
# Python dev
# hadolint ignore=DL3018,DL3059
RUN apk add --no-cache python-$PYTHON_VERSION-dev \
    python-$PYTHON_VERSION-base-dev \
    py$PYTHON_VERSION-pip-base \
    py$PYTHON_VERSION-setuptools

FROM chainguard_python_dev AS base

ARG UNAME
ARG UID
ARG GID
ARG WORKDIR
ARG AGENTDIR
ARG VENV_PATH

USER root

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# hadolint ignore=DL3018,DL3059
RUN apk add --no-cache uv graphviz openblas openssh-server gzip zip unzip curl \
  openjdk-11 vim nano procps tzdata poppler-utils
# hadolint ignore=DL3018,DL3059
RUN apk add --no-cache rust sqlite

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VENV_PATH=${VENV_PATH} \
    PIP_NO_CACHE_DIR=1 \
    NOTEBOOKS_KERNEL="python" \
    DEEPEVAL_HOME=/tmp/.deepeval \
    DEEPEVAL_TELEMETRY_OPT_OUT="YES"

ENV PATH="$VENV_PATH/bin:$PATH" \
  PYTHONPATH="/home/notebooks/.ipython/extensions:/home/notebooks/storage"

# hadolint ignore=SC1091
RUN uv venv ${VENV_PATH} && \
    source ${VENV_PATH}/bin/activate && \
    uv pip install -U pip setuptools
WORKDIR ${WORKDIR}

COPY ./agent/agent.py ./agent/cgroup_watchers.py ${AGENTDIR}/
COPY ./jupyter_kernel_gateway_config.py ./start_server.sh ${WORKDIR}/
COPY ./ipython_config.py /etc/ipython/
COPY ./extensions /etc/ipython/extensions

# Adding SSHD requirements
COPY ./sshd_config /etc/ssh/
RUN cp -a /etc/ssh /etc/ssh.cache && rm -rf /var/lib/apt/lists/*
RUN mkdir /etc/authorized_keys

# Custom user to run the image from
RUN addgroup -g "$GID" "$UNAME" && \
    adduser -D -u "$UID" -G "$UNAME" -s /bin/bash "$UNAME"

# Prompt customizations
COPY ./setup-prompt.sh /etc/profile.d/setup-prompt.sh

# additional setup scripts
COPY ./setup-ssh.sh ./common-user-limits.sh ./setup-venv.sh ${WORKDIR}/

# Adding SSHD requirements
RUN chown -R $UNAME:$UNAME ${WORKDIR} ${VENV_PATH} /home/notebooks /etc/ssh /etc/authorized_keys \
  # sshd prep
  && touch /etc/profile.d/notebooks-load-env.sh \
  && chown -R $UNAME:$UNAME /etc/profile.d/notebooks-load-env.sh \
  # Limit max processes
  && touch /etc/profile.d/bash-profile-load.sh \
  && chown -R $UNAME:$UNAME /etc/profile.d/bash-profile-load.sh

USER $UNAME

# Jupyter Gateway port
EXPOSE 8888
# sshd port
EXPOSE 22

FROM base AS builder
# this stage has only bare minimal of dependencies installed to optimize build time for the local development

ARG WORKDIR
ARG VENV_PATH

COPY ./requirements.txt ${WORKDIR}/
COPY ./agent/requirements-agent.txt ${WORKDIR}/

# hadolint ignore=SC1091
RUN source ${VENV_PATH}/bin/activate && \
    uv pip install --no-cache-dir -r ${WORKDIR}/requirements.txt && rm ${WORKDIR}/requirements.txt && \
    uv pip install --no-cache-dir -r ${WORKDIR}/requirements-agent.txt && rm ${WORKDIR}/requirements-agent.txt

# Copy agent runtime into work directory
COPY ./run_agent.py ${WORKDIR}/

RUN rm ${VENV_PATH}/share/jupyter/kernels/python3/kernel.json && chmod a+x ${WORKDIR}/start_server.sh
COPY ./kernel.json ${VENV_PATH}/share/jupyter/kernels/python3/

# Monitoring agent port
EXPOSE 8889

FROM base AS kernel
# this stage is what actually going to be run as kernel image and it's clean from all build junks

ARG UNAME
ARG WORKDIR
ARG GIT_COMMIT

LABEL com.datarobot.repo-name="notebooks"
LABEL com.datarobot.repo-sha=$GIT_COMMIT

RUN chown -R $UNAME:$UNAME ${WORKDIR} /home/notebooks

COPY --from=builder --chown=$UNAME $WORKDIR $WORKDIR

# This is required for custom models to work with this image
COPY ./start_server_drum.sh /opt/code/start_server.sh
ENV HOME=/opt CODE_DIR=/opt/code ADDRESS=0.0.0.0:8080
