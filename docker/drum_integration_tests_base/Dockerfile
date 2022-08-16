FROM rocker/r-apt:bionic
RUN echo '13 Apr 2021'
ENV LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install --no-install-recommends -y \
        apt-utils \
        curl \
        gpg-agent \
        software-properties-common \
        dirmngr \
        libssl-dev \
        ca-certificates \
        locales \
        libcurl4-openssl-dev \
        libxml2-dev \
        python3-pip \
        python3-dev \
        openjdk-11-jdk \
        openjdk-11-jre \
        libgomp1 \
        gcc \
        libc6-dev \
        pandoc \
        nginx \
        git \
        maven \
        docker.io && \
    rm -rf /var/lib/apt/lists/*

RUN echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen && \
    locale-gen en_US.utf8 && \
    /usr/sbin/update-locale LANG=en_US.UTF-8
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 && \
    add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/'

RUN apt-get install -y --no-install-recommends \
        r-cran-littler \
        r-base \
        r-base-dev \
        r-cran-recipes \
        r-cran-glmnet \
        r-cran-rjson \
        r-cran-e1071 \
        r-cran-plumber \
        r-cran-caret \
        r-cran-tidyverse \
        r-cran-pack \
        r-cran-rook && \
    rm -rf /var/lib/apt/lists/

RUN Rscript -e "install.packages(c('gert', 'usethis', 'devtools'), Ncpus=4)"

### Save cran as the default repo for R packages
RUN echo "r <- getOption('repos'); r['CRAN'] <- 'http://cran.rstudio.com'; options(repos = r);" > ~/.Rprofile

# Install caret models
RUN Rscript -e 'library(caret); install.packages(unique(modelLookup()[modelLookup()$forReg, c(1)]), Ncpus=4)'
RUN Rscript -e 'library(caret); install.packages(unique(modelLookup()[modelLookup()$forClass, c(1)]), Ncpus=4)'

RUN rm -rf /tmp/downloaded_packages/ /tmp/*.rds

## Install python 3.7.10 from source with --enabled-shared flag.
## In this case python will be dynamically linked to libpython.so, which is required for Julia.
## When python is installed as a package it is statically linked to libpython.a, which Julia doesn't like.
RUN apt update -y && apt install -y software-properties-common libsqlite3-dev libffi-dev && \
    add-apt-repository ppa:deadsnakes/ppa && \
    wget https://www.python.org/ftp/python/3.7.10/Python-3.7.10.tgz && \
    tar zxvf Python-3.7.10.tgz && \
    cd Python-3.7.10 && \
    ./configure --enable-shared --enable-loadable-sqlite-extensions && \
    make install
# By default, `make install' will install all the files in
# `/usr/local/bin', `/usr/local/lib' etc. You can specify
# an installation prefix other than `/usr/local' using `--prefix',
# for instance `--prefix=$HOME'.
RUN apt install -y python3.7-dev
# Create virtual env
RUN python3.7 -m pip install -U pip && \
    python3.7 -m pip install setuptools wheel virtualenv && \
    python3.7 -m virtualenv /opt/v3.7

ENV PYTHON=/opt/v3.7/bin/python

COPY requirements_drum.txt requirements_dropin.txt /tmp/

# uwsgi is an extra dependency and it shouldn't be in requirements_drum.txt, so install it explicitly
RUN cd /opt && . v3.7/bin/activate && \
    pip3 install -r /tmp/requirements_drum.txt -r /tmp/requirements_dropin.txt uwsgi && \
    rm -rf /root/.cache/pip

# # install julia 1.5.4
ENV HOME /opt
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.5/julia-1.5.4-linux-x86_64.tar.gz && \
    tar zxvf julia-1.5.4-linux-x86_64.tar.gz && \
    mkdir $HOME/.julia && \
    mkdir $HOME/julia

COPY sysim.jl $HOME/julia/sysim.jl
ENV JULIA_HOME=/julia-1.5.4
ENV JULIA_SYS_IMAGE=$HOME/julia/sys.so JULIA_PROJECT=$HOME/julia PATH=$PATH:$JULIA_HOME/bin

RUN julia "$HOME/julia/sysim.jl" && \
    chmod -R 777 $HOME/.julia/logs/manifest_usage.toml && \
    rm julia-1.5.4-linux-x86_64.tar.gz && \
    rm Python-3.7.10.tgz

USER root