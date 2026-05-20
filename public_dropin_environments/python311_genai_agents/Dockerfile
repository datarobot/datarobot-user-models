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
# Writable at build and runtime (custom model / uv); override with --build-arg if needed.
ARG UV_CACHE_DIR=/tmp/uv-cache

FROM datarobot/mirror_chainguard_datarobot.com_python-fips:3.11-dev AS base

ARG UNAME
ARG UID
ARG GID
ARG WORKDIR
ARG AGENTDIR
ARG VENV_PATH
ARG UV_CACHE_DIR

USER root

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# hadolint ignore=DL3018,DL3059
RUN apk add --no-cache uv graphviz openblas openssh-server \
    gzip zip unzip curl \
    openjdk-11 vim nano procps tzdata
# hadolint ignore=DL3018,DL3059
RUN apk add --no-cache rust sqlite

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPYCACHEPREFIX=/tmp/.python_cache \
    UV_COMPILE_BYTECODE=1 \
    VENV_PATH=${VENV_PATH} \
    PIP_NO_CACHE_DIR=1 \
    NOTEBOOKS_KERNEL="python" \
    DEEPEVAL_HOME=/tmp/.deepeval \
    DEEPEVAL_TELEMETRY_OPT_OUT="YES"

ENV PATH="$VENV_PATH/bin:$PATH" \
    PYTHONPATH="/home/notebooks/.ipython/extensions:/home/notebooks/storage"

RUN uv venv ${VENV_PATH}
WORKDIR ${WORKDIR}

COPY ./agent/agent.py ./agent/cgroup_watchers.py ${AGENTDIR}/
COPY ./jupyter_kernel_gateway_config.py ${WORKDIR}/jupyter_kernel_gateway_config.py
COPY ./start_server_codespaces.sh ${WORKDIR}/start_server.sh
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
  && chown -R $UNAME:$UNAME /etc/profile.d/bash-profile-load.sh \
  && mkdir -p "${UV_CACHE_DIR}" \
  && chown -R "$UNAME:$UNAME" "${UV_CACHE_DIR}"

# Set after root `uv venv` / chown so Kaniko/BuildKit: `uv sync` as notebooks does not inherit a
# root-owned ${UV_CACHE_DIR} (see sdists-v9/.git permission denied).
ENV UV_CACHE_DIR=${UV_CACHE_DIR}

USER $UNAME

# Jupyter Gateway port
EXPOSE 8888
# sshd port
EXPOSE 22

FROM base AS builder
# this stage has only bare minimal of dependencies installed to optimize build time for the local development

ENV UV_COMPILE_BYTECODE=1

ARG WORKDIR
ARG VENV_PATH
ARG UNAME

COPY ./pyproject.toml ${WORKDIR}/pyproject.toml
COPY ./uv.lock ${WORKDIR}/uv.lock

WORKDIR ${WORKDIR}
USER $UNAME

RUN . ${VENV_PATH}/bin/activate && \
    uv sync --frozen --extra agentic_playground && \
    # Cleanup after nemoguardrails (.vscode, vscode_extension, .venv installed)..
    rm -rf ${VENV_PATH}/lib/python3.11/site-packages/.vscode ${VENV_PATH}/lib/python3.11/site-packages/vscode_extension ${VENV_PATH}/lib/python3.11/site-packages/.venv && \
    chmod -R 777 ${VENV_PATH}
WORKDIR /

# Copy agent runtime into work directory
COPY ./run_agent.py ${WORKDIR}/

RUN rm ${VENV_PATH}/share/jupyter/kernels/python3/kernel.json && chmod a+x ${WORKDIR}/start_server.sh
COPY ./kernel.json ${VENV_PATH}/share/jupyter/kernels/python3/

# Monitoring agent port
EXPOSE 8889

# CMPT-5321 / Kaniko: continue from builder so the kernel .venv is not re-tarred across
# COPY --from=builder (snapshot symlink / ownership issues).
FROM builder AS kernel
# this stage is what actually going to be run as kernel image and it's clean from all build junks

ARG UNAME=notebooks
ARG GIT_COMMIT
ARG VENV_PATH
ARG UV_CACHE_DIR=/tmp/uv-cache

LABEL com.datarobot.repo-name="notebooks"
LABEL com.datarobot.repo-sha=$GIT_COMMIT

USER root
RUN mkdir -p /opt/venv && chmod a+rwx /opt/venv
USER $UNAME

# This is required for custom models to work with this image
COPY ./start_server_custom_model.sh /opt/code/start_server.sh
ENV VENV_DIR=/opt/venv \
    UV_CACHE_DIR=${UV_CACHE_DIR} \
    UV_LINK_MODE=copy \
    UV_FROZEN=1 \
    HOME=/opt \
    CODE_DIR=/opt/code \
    ADDRESS=0.0.0.0:8080
