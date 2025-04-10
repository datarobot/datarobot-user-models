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
ARG R_VERSION=4.3.1

ARG UNAME=notebooks
ARG UID=10101
ARG GID=10101

FROM golang:1.22.8 as ghcli_builder

WORKDIR /workspace

RUN git clone https://github.com/cli/cli.git gh-cli

WORKDIR /workspace/gh-cli

RUN git checkout v2.58.0  \
    && CGO_ENABLED=0 make install prefix=/workspace/gh

FROM registry.access.redhat.com/ubi8/ubi-minimal:8.10 AS base
ARG WORKDIR
ARG R_VERSION
ARG AGENTDIR

ARG UNAME
ARG UID
ARG GID

ENV PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="/home/notebooks/storage" \
  R_LIBS_USER=${WORKDIR}/renv \
  NOTEBOOKS_KERNEL="r"

EXPOSE 8888
EXPOSE 8889
# sshd port
EXPOSE 22

WORKDIR ${WORKDIR}

COPY ./.Rprofile ${WORKDIR}/
COPY ./jupyter_kernel_gateway_config.py ./start_server.sh ${WORKDIR}/
COPY ./dr_requirements.txt ./agent/requirements-agent.txt ${WORKDIR}/
COPY ./setup-base.R ${WORKDIR}/
COPY ./agent/agent.py ./agent/cgroup_watchers.py ${AGENTDIR}/

# Shell customizations
COPY ./setup-shell.sh /etc/profile.d/setup-shell.sh

# additional setup scripts
COPY ./setup-ssh.sh ./common-user-limits.sh ./setup-venv.sh ${WORKDIR}/

# Set the SHELL option -o pipefail before RUN with a pipe in it.
# Rationale: https://github.com/hadolint/hadolint/wiki/DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN microdnf update -y \
  && microdnf install -y tar gzip python39 bzip2-devel cmake gcc gcc-c++  \
  gcc-gfortran libSM libXmu libXt libcurl-devel libicu-devel libtiff  \
  make openblas-threads pango pcre2-devel tcl tk unzip which \
  xz-devel zip zlib-devel libxml2-devel openssl-devel libpng-devel \
  libjpeg-turbo-devel shadow-utils git wget fontconfig-devel glib2-devel  \
  libtiff-devel vim-minimal nano tzdata-java mesa-libGLU harfbuzz \
  && microdnf install -y openssh-server \
  && microdnf clean all

# Requirements to install 'devtools'
RUN curl -O https://vault.centos.org/8-stream/AppStream/x86_64/os/Packages/graphite2-devel-1.3.10-10.el8.x86_64.rpm  \
  && rpm -i graphite2-devel-1.3.10-10.el8.x86_64.rpm  \
  && rm graphite2-devel-1.3.10-10.el8.x86_64.rpm
RUN curl -O https://vault.centos.org/8-stream/AppStream/x86_64/os/Packages/harfbuzz-icu-1.7.5-4.el8.x86_64.rpm  \
  && rpm -i harfbuzz-icu-1.7.5-4.el8.x86_64.rpm  \
  && rm harfbuzz-icu-1.7.5-4.el8.x86_64.rpm
RUN curl -O https://vault.centos.org/8-stream/AppStream/x86_64/os/Packages/harfbuzz-devel-1.7.5-4.el8.x86_64.rpm  \
  && rpm -i harfbuzz-devel-1.7.5-4.el8.x86_64.rpm  \
  && rm harfbuzz-devel-1.7.5-4.el8.x86_64.rpm
RUN curl -O https://vault.centos.org/8-stream/AppStream/x86_64/os/Packages/fribidi-devel-1.0.4-9.el8.x86_64.rpm  \
  && rpm -i fribidi-devel-1.0.4-9.el8.x86_64.rpm  \
  && rm fribidi-devel-1.0.4-9.el8.x86_64.rpm


RUN curl -O https://cdn.rstudio.com/r/centos-8/pkgs/R-${R_VERSION}-1-1.x86_64.rpm \
  && rpm -i R-${R_VERSION}-1-1.x86_64.rpm \
  && rm R-${R_VERSION}-1-1.x86_64.rpm \
  && ln -s /opt/R/${R_VERSION}/bin/R /usr/local/bin/R \
  && ln -s /opt/R/${R_VERSION}/bin/Rscript /usr/local/bin/Rscript

RUN microdnf install -y java-1.8.0-openjdk-devel java-11-openjdk-devel && R CMD javareconf && microdnf clean all

RUN pip3 install --no-cache-dir -r ${WORKDIR}/dr_requirements.txt \
  && pip3 install --no-cache-dir -r ${WORKDIR}/requirements-agent.txt \
  && rm ${WORKDIR}/requirements-agent.txt \
  && rm ${WORKDIR}/dr_requirements.txt \
  && chmod a+x ${WORKDIR}/start_server.sh

RUN R < setup-base.R --no-save && rm ${WORKDIR}/setup-base.R

RUN mkdir ${WORKDIR}/renv

FROM base AS kernel-prep

ARG WORKDIR
ARG UNAME
ARG UID
ARG GID

RUN groupadd -g $GID -o $UNAME \
  && useradd -l -m -u $UID -g $GID -o -s /bin/bash $UNAME

RUN curl https://github.com/jgm/pandoc/releases/download/2.18/pandoc-2.18-linux-amd64.tar.gz -o pandoc-2.18-linux-amd64.tar.gz -L \
  && mkdir /usr/local/pandoc \
  && tar xvzf pandoc-2.18-linux-amd64.tar.gz --strip-components 1 -C /usr/local/pandoc \
  && ln -s /usr/local/pandoc/bin/pandoc /usr/local/bin/pandoc \
  && rm pandoc-2.18-linux-amd64.tar.gz

RUN R -e 'install.packages("renv")'
COPY ./renv.lock ${WORKDIR}/

COPY --from=base --chown=$UNAME $WORKDIR $WORKDIR

# Adding SSHD requirements
COPY ./sshd_config /etc/ssh/
RUN cp -a /etc/ssh /etc/ssh.cache \
  && rm -rf /var/cache/apk/* \
  && mkdir /etc/authorized_keys \
  && chown -R $UNAME:$UNAME ${WORKDIR} /home/notebooks /etc/ssh /etc/authorized_keys \
  # sshd prep
  && touch /etc/profile.d/notebooks-load-env.sh \
  && chown -R $UNAME:$UNAME /etc/profile.d/notebooks-load-env.sh \
  # Limit max processes
  && touch /etc/profile.d/bash-profile-load.sh \
  && chown -R $UNAME:$UNAME /etc/profile.d/bash-profile-load.sh

USER $UNAME

RUN R -e 'renv::restore()' && rm renv.lock

FROM kernel-prep as kernel

ARG GIT_COMMIT

LABEL com.datarobot.repo-name="notebooks"
LABEL com.datarobot.repo-sha=$GIT_COMMIT

USER root

# remove microdnf
RUN microdnf remove microdnf

USER $UNAME

COPY --from=ghcli_builder --chown=$UNAME /workspace/gh/bin/gh /usr/bin/gh

FROM kernel-prep as devel
# This stage should only be used in local development

ARG GIT_COMMIT

LABEL com.datarobot.repo-name="notebooks"
LABEL com.datarobot.repo-sha=$GIT_COMMIT

USER root

# install things that devs would actually like to use, and possibly things
# we want to pull up a stage to the production kernel
RUN microdnf install -y procps-ng vim-enhanced \
  && microdnf clean all

USER $UNAME
