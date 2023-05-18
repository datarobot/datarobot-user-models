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
ARG R_VERSION=4.2.2

ARG UNAME=notebooks
ARG UID=90909
ARG GID=90909

FROM registry.access.redhat.com/ubi8/ubi-minimal:8.7 AS base
ARG WORKDIR
ARG R_VERSION
ARG AGENTDIR

ARG UNAME
ARG UID
ARG GID

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="/home/notebooks/storage" \
    R_LIBS_USER=${WORKDIR}/renv

EXPOSE 8888
EXPOSE 8889
# sshd port
EXPOSE 22

WORKDIR ${WORKDIR}

COPY ./.Rprofile ${WORKDIR}/
COPY ./jupyter_kernel_gateway_config.py ./start_server.sh ${WORKDIR}/
COPY ./agent/agent.py ${AGENTDIR}/
COPY ./agent/cgroup_monitor.py ${AGENTDIR}/
COPY ./dr_requirements.txt ./agent/requirements-agent.txt ${WORKDIR}/
COPY ./setup-common.R ${WORKDIR}/

# Prompt customizations
COPY ./setup-prompt.sh /etc/profile.d/setup-prompt.sh
# sshd prep
COPY ./setup-ssh.sh ${WORKDIR}/
# set user limits
COPY ./common-user-limits.sh ${WORKDIR}/

# Add any package that will be installed on system level here:
RUN microdnf update \
  && microdnf install -y tar-2:1.30 gzip-1.9 python39-3.9.16 bzip2-devel-1.0.6 cmake-3.20.2 gcc-8.5.0 gcc-c++-8.5.0  \
    gcc-gfortran-8.5.0 libSM-1.2.3 libXmu-1.1.3 libXt-1.1.5 libcurl-devel-7.61.1 libicu-devel-60.3 libtiff-4.0.9  \
    make-1:4.2.1 openblas-threads-0.3.15 pango-1.42.4 pcre2-devel-10.32 tcl-1:8.6.8 tk-1:8.6.8 unzip-6.0 which-2.21 \
    xz-devel-5.2.4 zip-3.0 zlib-devel-1.2.11 libxml2-devel-2.9.7 openssl-devel-1:1.1.1k libpng-devel-2:1.6.34 \
  libjpeg-turbo-devel-1.5.3 shadow-utils-2:4.6 git-2.39.1 wget-1.19.5 fontconfig-devel-2.13.1 glib2-devel-2.56.4  \
  libtiff-devel-4.0.9 vim-minimal-2:8.0.1763 nano-2.9.8 \
  && microdnf install -y openssh-server \
    && microdnf clean all

# Requirements to install 'devtools'
RUN curl -O http://mirror.centos.org/centos/8-stream/AppStream/x86_64/os/Packages/graphite2-devel-1.3.10-10.el8.x86_64.rpm  \
  && rpm -i graphite2-devel-1.3.10-10.el8.x86_64.rpm  \
  && rm graphite2-devel-1.3.10-10.el8.x86_64.rpm
RUN curl -O http://mirror.centos.org/centos/8-stream/AppStream/x86_64/os/Packages/harfbuzz-icu-1.7.5-3.el8.x86_64.rpm  \
  && rpm -i harfbuzz-icu-1.7.5-3.el8.x86_64.rpm  \
  && rm harfbuzz-icu-1.7.5-3.el8.x86_64.rpm
RUN curl -O https://vault.centos.org/centos/8/AppStream/x86_64/os/Packages/harfbuzz-devel-1.7.5-3.el8.x86_64.rpm  \
  && rpm -i harfbuzz-devel-1.7.5-3.el8.x86_64.rpm  \
  && rm harfbuzz-devel-1.7.5-3.el8.x86_64.rpm
RUN curl -O http://mirror.centos.org/centos/8-stream/AppStream/x86_64/os/Packages/fribidi-devel-1.0.4-9.el8.x86_64.rpm  \
  && rpm -i fribidi-devel-1.0.4-9.el8.x86_64.rpm  \
  && rm fribidi-devel-1.0.4-9.el8.x86_64.rpm


RUN curl -O https://cdn.rstudio.com/r/centos-8/pkgs/R-${R_VERSION}-1-1.x86_64.rpm \
  && rpm -i R-${R_VERSION}-1-1.x86_64.rpm \
  && rm R-${R_VERSION}-1-1.x86_64.rpm \
  && ln -s /opt/R/${R_VERSION}/bin/R /usr/local/bin/R \
  && ln -s /opt/R/${R_VERSION}/bin/Rscript /usr/local/bin/Rscript

RUN microdnf install -y java-11-openjdk-devel && R CMD javareconf && microdnf clean all

RUN pip3 install --no-cache-dir -r ${WORKDIR}/dr_requirements.txt \
  && pip3 install --no-cache-dir -r ${WORKDIR}/requirements-agent.txt \
  && rm ${WORKDIR}/requirements-agent.txt \
  && rm ${WORKDIR}/dr_requirements.txt \
  && chmod a+x ${WORKDIR}/start_server.sh

RUN R < setup-common.R --no-save && rm ${WORKDIR}/setup-common.R


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

RUN R -e 'renv::restore()' && rm renv.lock \
  # Restore the R environment from lockfile, then
  # Remove files that are marked by trivy as vulnerabilities
  && rm -rf /home/notebooks/.cache/R/renv/cache/v5/R-4.2/x86_64-pc-linux-gnu/sass/0.4.2/1b191143d7d3444d504277843f3a95fe/sass/doc/sass.html \
  && rm -rf $WORKDIR/renv/survminer/doc/Playing_with_fonts_and_texts.html \
  && rm -rf /home/notebooks/.cache/R/renv/cache/v5/R-4.2/x86_64-pc-linux-gnu/openssl/2.0.4/e86c5ffeb8474a9e03d75f5d2919683e/openssl/doc/keys.html \
  && rm -rf /home/notebooks/.cache/R/renv/cache/v5/R-4.2/x86_64-pc-linux-gnu/openssl/2.0.4/e86c5ffeb8474a9e03d75f5d2919683e/openssl/doc/secure_rng.html \
  && rm -rf /home/notebooks/.cache/R/renv/cache/v5/R-4.2/x86_64-pc-linux-gnu/survminer/0.4.9/3f29f006a8eb499eff91d8b72325756e/survminer/doc/Specifiying_weights_in_log-rank_comparisons.html \
  && rm -rf /home/notebooks/.cache/R/renv/cache/v5/R-4.2/x86_64-pc-linux-gnu/survminer/0.4.9/3f29f006a8eb499eff91d8b72325756e/survminer/doc/Playing_with_fonts_and_texts.html \
  && rm -rf $WORKDIR/renv/openssl/doc/keys.html \
  && rm -rf $WORKDIR/renv/sass/doc/sass.html \
  && rm -rf $WORKDIR/renv/survminer/doc/Specifiying_weights_in_log-rank_comparisons.html \
  && rm -rf $WORKDIR/renv/openssl/doc/secure_rng.html


FROM kernel-prep as kernel

ARG GIT_COMMIT

LABEL com.datarobot.repo-name="notebooks"
LABEL com.datarobot.repo-sha=$GIT_COMMIT

USER root

# remove microdnf
RUN microdnf remove microdnf

USER $UNAME

FROM kernel-prep as devel
# This stage should only be used in local development

ARG GIT_COMMIT

LABEL com.datarobot.repo-name="notebooks"
LABEL com.datarobot.repo-sha=$GIT_COMMIT

USER root

# install things that devs would actually like to use, and possibly things
# we want to pull up a stage to the production kernel
RUN microdnf install -y procps-ng-3.3.15 vim-enhanced-2:8.0.1763 \
  && microdnf clean all

USER $UNAME
