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
#   that will be installed through yum or            #
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

ARG PYTHON_VERSION=3
ARG PYTHON_EXACT_VERSION=3.9.18

FROM golang:1.22.8 as ghcli_builder

WORKDIR /workspace

RUN git clone https://github.com/cli/cli.git gh-cli

WORKDIR /workspace/gh-cli

RUN git checkout v2.58.0  \
    && CGO_ENABLED=0 make install prefix=/workspace/gh


FROM registry.access.redhat.com/ubi8/ubi-minimal:8.10 AS pulumi_builder
RUN microdnf update \
  && microdnf install -y tar gzip \
  && microdnf clean all

RUN curl -o install_pulumi.sh https://get.pulumi.com && chmod +x install_pulumi.sh
RUN ./install_pulumi.sh --version latest --install-root ./pulumi --no-edit-path

FROM nvidia/cuda:12.3.1-devel-ubi9 AS base
# some globally required dependencies

ARG UNAME
ARG UID
ARG GID
ARG WORKDIR
ARG AGENTDIR
ARG VENV_PATH
ARG PYTHON_VERSION
ARG PYTHON_EXACT_VERSION

# Set the SHELL option -o pipefail before RUN with a pipe in it.
# Rationale: https://github.com/hadolint/hadolint/wiki/DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Add any package that will be installed on system level here:
RUN yum update -y \
  && yum install -y python$PYTHON_VERSION-$PYTHON_EXACT_VERSION python$PYTHON_VERSION-devel-$PYTHON_EXACT_VERSION \
  gcc-11.4.1 gcc-c++-11.4.1 glibc-devel-2.34 libffi-devel-3.4.2 python3-pip-21.2.3 \
  openblas-0.3.21 python$PYTHON_VERSION-scipy shadow-utils-2:4.9 passwd-0.80 git-2.43.5 openssh-server tar-2:1.34 gzip-1.12 unzip-6.0 zip-3.0 wget-1.21.1 \
  java-11-openjdk-headless-1:11.0.25.0.9-2.el9 vim-minimal-2:8.2.2637 nano-5.6.1 procps-ng-3.3.17 tzdata-java poppler-utils-21.01.0 \
  && yum clean all

ENV PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  VENV_PATH=${VENV_PATH} \
  PIP_NO_CACHE_DIR=1 \
  NOTEBOOKS_KERNEL="python"

ENV PATH="$VENV_PATH/bin:$PATH" \
  PYTHONPATH="/home/notebooks/.ipython/extensions:/home/notebooks/storage"

RUN python3 -m venv ${VENV_PATH} && pip3 install -U pip setuptools
WORKDIR ${WORKDIR}

COPY ./agent/agent.py ./agent/cgroup_watchers.py ${AGENTDIR}/
COPY ./jupyter_kernel_gateway_config.py ./start_server.sh ${WORKDIR}/
COPY ./ipython_config.py /etc/ipython/
COPY ./extensions /etc/ipython/extensions

# Adding SSHD requirements
COPY ./sshd_config /etc/ssh/
RUN cp -a /etc/ssh /etc/ssh.cache && rm -rf /var/cache/apk/*
RUN mkdir /etc/authorized_keys

# Removing pip leftovers to not have trivy complain
RUN find /lib/python3.*/site-packages -name "pip-*.dist-info" -type d -exec rm -rf {} + \
  && find /lib/python3.*/site-packages -name "setuptools-*.dist-info" -type d -exec rm -rf {} +

# Custom user to run the image from

RUN groupadd -g $GID -o $UNAME && \
  useradd -l -m -u $UID -g $GID -o -s /bin/bash $UNAME

# Shell customizations
COPY ./setup-shell.sh /etc/profile.d/setup-shell.sh

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

COPY ./dr_requirements.txt ./agent/requirements-agent.txt ./requirements-gpu.txt ${WORKDIR}/
RUN pip3 install --no-cache-dir -r ${WORKDIR}/dr_requirements.txt \
  && pip3 install --no-cache-dir -r ${WORKDIR}/requirements-agent.txt \
  && pip3 install --no-cache-dir --extra-index-url https://pypi.nvidia.com -r ${WORKDIR}/requirements-gpu.txt \
  && rm ${WORKDIR}/dr_requirements.txt \
  && rm ${WORKDIR}/requirements-gpu.txt \
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
COPY ./requirements-gpu.txt ${WORKDIR}/
RUN pip3 install --use-deprecated=legacy-resolver --no-cache-dir -r ${WORKDIR}/requirements.txt \
  && pip3 install --no-cache-dir -r ${WORKDIR}/requirements-gpu.txt \
  && rm ${WORKDIR}/requirements.txt && rm ${WORKDIR}/requirements-gpu.txt

FROM base AS kernel

# this stage is what actually going to be run as kernel image and it's clean from all build junks

ARG UNAME

ARG WORKDIR

ARG GIT_COMMIT

LABEL com.datarobot.repo-name="notebooks"
LABEL com.datarobot.repo-sha=$GIT_COMMIT

RUN chown -R $UNAME:$UNAME ${WORKDIR} /home/notebooks

COPY --from=builder --chown=$UNAME $WORKDIR $WORKDIR
COPY --from=ghcli_builder --chown=$UNAME /workspace/gh/bin/gh /usr/bin/gh
COPY --from=pulumi_builder --chown=$UNAME /pulumi/bin /usr/bin/

# Removing pip leftovers to not have trivy complain
RUN find "${VENV_PATH}"/lib/python3.*/site-packages -name "pip-*.dist-info" -type d -exec rm -rf {} + \
  && find "${VENV_PATH}"/lib/python3.*/site-packages -name "setuptools-*.dist-info" -type d -exec rm -rf {} +
