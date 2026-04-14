# Copyright 2024 DataRobot, Inc. and its affiliates.
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

# This is a private production chain-guard image that is stored in DataRobot's private registry.
# Replace it with your own production chain-guard image if you build your own.
FROM datarobot/mirror_chainguard_datarobot.com_python-fips:3.11-dev AS base

ARG UNAME
ARG UID
ARG GID
ARG WORKDIR
ARG AGENTDIR
ARG VENV_PATH
ARG GIT_COMMIT

# Set the SHELL option -o pipefail before RUN with a pipe in it.
# Rationale: https://github.com/hadolint/hadolint/wiki/DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root
RUN echo "[Python3.11] GIT_COMMIT: $GIT_COMMIT" && \
  apk update && apk add --no-cache gcc libstdc++ glibc-dev libffi-dev graphviz \
  openblas git openssh-server gzip zip unzip wget curl openjdk-11 vim nano \
  && rm -rf /var/cache/apk/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VENV_PATH=${VENV_PATH} \
    PIP_NO_CACHE_DIR=1 \
    NOTEBOOKS_KERNEL="python"

ENV PATH="$VENV_PATH/bin:$PATH" \
  PYTHONPATH="/home/notebooks/.ipython/extensions:/home/notebooks/storage"

RUN python3 -m venv ${VENV_PATH} && pip3 install -U pip setuptools
WORKDIR ${WORKDIR}

# Install git helper binary used for private git authentication in Notebooks/Codepaces
RUN curl -L -o drgithelper https://github.com/datarobot-oss/drgithelper/releases/download/v0.0.24/drgithelper && chmod +x drgithelper

COPY ./agent/agent.py ./agent/cgroup_watchers.py ${AGENTDIR}/
COPY ./jupyter_kernel_gateway_config.py ./start_server.sh ${WORKDIR}/
COPY ./ipython_config.py /etc/ipython/
COPY ./extensions /etc/ipython/extensions

# Adding SSHD requirements
COPY ./sshd_config /etc/ssh/
RUN cp -a /etc/ssh /etc/ssh.cache && rm -rf /var/cache/apk/*
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

FROM base AS minimal
# this stage has only bare minimum of dependencies installed to optimize build time for local development

ARG WORKDIR
ARG VENV_PATH

COPY ./dr_requirements.txt ./agent/requirements-agent.txt ${WORKDIR}/
RUN python3 -m pip install --no-cache-dir -r ${WORKDIR}/dr_requirements.txt \
    && python3 -m pip install --no-cache-dir -r ${WORKDIR}/requirements-agent.txt \
    && rm ${WORKDIR}/dr_requirements.txt \
    && rm ${WORKDIR}/requirements-agent.txt \
    && rm ${VENV_PATH}/share/jupyter/kernels/python3/kernel.json \
    && chmod a+x ${WORKDIR}/start_server.sh

# Monitoring agent port
EXPOSE 8889

FROM minimal AS builder
# this stage includes all data science dependencies we want to have in the kernel runtime out of the box

ARG WORKDIR
ARG VENV_PATH

COPY ./kernel.json ${VENV_PATH}/share/jupyter/kernels/python3/
COPY ./requirements.txt ${WORKDIR}/
RUN pip3 install --no-cache-dir -r ${WORKDIR}/requirements.txt \
  && rm ${WORKDIR}/requirements.txt

FROM base AS kernel
# this stage is what is actually going to be run as kernel image and it's clean from all build junk

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
