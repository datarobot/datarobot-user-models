# Copyright 2023 DataRobot, Inc. and its affiliates.
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

# You can specify a different python version here but be sure that
# it's available in microdnf repo
# to check use this bash commands:
# ```bash```
# docker run --rm -it registry.access.redhat.com/ubi8/ubi-minimal:8.7 bash
# microdnf repoquery python3*
# ```
ARG PYTHON_VERSION=38
ARG PYTHON_EXACT_VERSION=3.8.16

FROM registry.access.redhat.com/ubi8/ubi-minimal:8.7 AS base
# some globally required dependencies

ARG UNAME
ARG UID
ARG GID
ARG WORKDIR
ARG AGENTDIR
ARG VENV_PATH
ARG PYTHON_VERSION
ARG PYTHON_EXACT_VERSION

# Add any package that will be installed on system level here:
RUN microdnf update \
  && microdnf install -y python$PYTHON_VERSION-$PYTHON_EXACT_VERSION python$PYTHON_VERSION-devel-$PYTHON_EXACT_VERSION \
  gcc-8.5.0 gcc-c++-8.5.0 glibc-devel-2.28 libffi-devel-3.1 graphviz-2.40.1 \
  openblas-0.3.15 python$PYTHON_VERSION-scipy shadow-utils-2:4.6 passwd-0.80 git-2.39.3 openssh-server tar-2:1.30 gzip-1.9 unzip-6.0 zip-3.0 wget-1.19.5 \
  java-11-openjdk-headless-1:11.0.20.0.8-3.el8.x86_64 vim-minimal-2:8.0.1763-19.el8_6.4.x86_64 nano-2.9.8 \
  && pip3 install -U --no-cache-dir pip==23.1.2 \
  && microdnf clean all

RUN pip3 install --no-cache-dir pip==23.1.2 setuptools==66.0.0

ENV PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  VENV_PATH=${VENV_PATH}

ENV PATH="$VENV_PATH/bin:$PATH" \
  PYTHONPATH="/home/notebooks/.ipython/extensions:/home/notebooks/storage"

RUN python3 -m venv ${VENV_PATH}
WORKDIR ${WORKDIR}

COPY ./agent/agent.py ./agent/cgroup_watchers.py ${AGENTDIR}/
COPY ./extensions /home/notebooks/.ipython/extensions
COPY ./jupyter_kernel_gateway_config.py ./start_server.sh ${WORKDIR}/
COPY ./ipython_config.py /etc/ipython/

# Adding SSHD requirements
COPY ./sshd_config /etc/ssh/
RUN cp -a /etc/ssh /etc/ssh.cache && rm -rf /var/cache/apk/*
RUN mkdir /etc/authorized_keys


# Removing pip leftovers to not have trivy complain
RUN rm -rf /lib/python3.9/site-packages/pip-20.2.4.dist-info && \
  rm -rf /nbx/.venv/lib/python3.9/site-packages/pip-20.2.4.dist-info && \
  rm -rf /lib/python3.8/site-packages/pip-19.3.1.dist-info && \
  rm -rf /nbx/.venv/lib/python3.8/site-packages/pip-19.3.1.dist-info

# Custom user to run the image from

RUN groupadd -g $GID -o $UNAME && \
  useradd -l -m -u $UID -g $GID -o -s /bin/bash $UNAME

# Prompt customizations
COPY ./setup-prompt.sh /etc/profile.d/setup-prompt.sh

# remove microdnf
RUN microdnf remove microdnf

# sshd prep
COPY ./setup-ssh.sh ${WORKDIR}/
# set user limits
COPY ./common-user-limits.sh ${WORKDIR}/

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
RUN pip3 install --no-cache-dir -r ${WORKDIR}/dr_requirements.txt \
  && pip3 install --no-cache-dir -r ${WORKDIR}/requirements-agent.txt \
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

# Removing pip leftovers to not have trivy complain
RUN rm -rf /lib/python3.9/site-packages/pip-20.2.4.dist-info && \
  rm -rf /nbx/.venv/lib/python3.9/site-packages/pip-20.2.4.dist-info && \
  rm -rf /lib/python3.8/site-packages/pip-19.3.1.dist-info && \
  rm -rf /nbx/.venv/lib/python3.8/site-packages/pip-19.3.1.dist-info && \
  rm -rf "${VENV_PATH}"/lib/python3.9/site-packages/setuptools-50.3.2.dist-info && \
  rm -rf "${VENV_PATH}"/lib/python3.8/site-packages/setuptools-41.6.0.dist-info

RUN chown -R $UNAME:$UNAME ${WORKDIR} /home/notebooks

COPY --from=builder --chown=$UNAME $WORKDIR $WORKDIR
