# Copyright 2024 DataRobot, Inc. and its affiliates.
# All rights reserved.
# DataRobot, Inc. Confidential.
# This is unpublished proprietary source code of DataRobot, Inc.
# and its affiliates.
# The copyright notice above does not evidence any actual or intended
# publication of such source code.


###################   !NOTA BENE!   #######################
#   All the files, parameters and packages are necessary  #
#   for the proper functioning of Notebooks.              #
#   If needed, you can include any system package         #
#   that will be installed through microdnf or            #
#   add a required package to the requirements.txt file.  #
#   Please note that removing predefined packages         #
#   may result in issues with Notebooks functionality.    #
###########################################################

ARG WORKDIR=/etc/system/kernel
ARG AGENTDIR=/etc/system/kernel/agent
ARG VENV_PATH=${WORKDIR}/.venv

ARG UNAME=notebooks
ARG UID=10101
ARG GID=10101

# You can specify a different python version here
# be sure that package available in microdnf repo
# to check use this bash commands:
# ```bash```
# docker run --rm -it registry.access.redhat.com/ubi9/ubi-minimal:9.5 bash
# microdnf repoquery python3*
# ```
ARG PYTHON_VERSION=3.11
ARG PYTHON_EXACT_VERSION=3.11.9

FROM registry.access.redhat.com/ubi9/ubi-minimal:9.5 AS base
# some globally required dependencies

ARG UNAME
ARG UID
ARG GID
ARG WORKDIR
ARG AGENTDIR
ARG VENV_PATH
ARG PYTHON_VERSION
ARG PYTHON_EXACT_VERSION
ARG GIT_COMMIT

# Set the SHELL option -o pipefail before RUN with a pipe in it.
# Rationale: https://github.com/hadolint/hadolint/wiki/DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Add any package that will be installed on system level here:
RUN echo "[Python$PYTHON_VERSION] GIT_COMMIT: $GIT_COMMIT" \
    && microdnf update -y \
    && microdnf install -y python$PYTHON_VERSION-$PYTHON_EXACT_VERSION python3-devel-3.9.21 \
  gcc-11.5.0 gcc-c++-11.5.0 glib2-devel-2.68.4 libffi-devel-3.4.2 graphviz-2.44.0 python$PYTHON_VERSION-pip \
  openblas-0.3.26 python$PYTHON_VERSION-scipy shadow-utils passwd git openssh-server tar gzip unzip zip wget \
  java-11-openjdk-headless vim-minimal nano \
  && microdnf clean all

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
RUN curl -L -o drgithelper https://github.com/datarobot-oss/drgithelper/releases/download/v0.0.9/drgithelper && chmod +x drgithelper

COPY ./agent/agent.py ./agent/cgroup_watchers.py ${AGENTDIR}/
COPY ./jupyter_kernel_gateway_config.py ./start_server.sh ${WORKDIR}/
COPY ./ipython_config.py /etc/ipython/
COPY ./extensions /etc/ipython/extensions

# Adding SSHD requirements
COPY ./sshd_config /etc/ssh/
RUN cp -a /etc/ssh /etc/ssh.cache && rm -rf /var/cache/apk/*
RUN mkdir /etc/authorized_keys

# Custom user to run the image from
RUN groupadd -g $GID -o $UNAME && \
    useradd -l -m -u $UID -g $GID -o -s /bin/bash $UNAME

# Prompt customizations
COPY ./setup-prompt.sh /etc/profile.d/setup-prompt.sh

# remove microdnf
RUN microdnf remove microdnf -y

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
# this stage has only bare minimal of dependencies installed to optimize build time for the local development

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
ARG PYTHON_VERSION

COPY ./kernel.json ${VENV_PATH}/share/jupyter/kernels/python3/
COPY ./requirements.txt ${WORKDIR}/
RUN pip3 install --no-cache-dir -r ${WORKDIR}/requirements.txt \
  && rm ${WORKDIR}/requirements.txt

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
