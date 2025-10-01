FROM ubuntu:24.04
ENV LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install --no-install-recommends -y \
        apt-utils \
        build-essential \
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
        python3-venv \
        openjdk-11-jdk \
        openjdk-11-jre \
        libgomp1 \
        gcc \
        libc6-dev \
        pandoc \
        git \
        maven \
        vim \
        docker.io && \
    rm -rf /var/lib/apt/lists/*

RUN echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen && \
    locale-gen en_US.utf8 && \
    /usr/sbin/update-locale LANG=en_US.UTF-8

ENV VIRTUAL_ENV=/opt/venv
COPY requirements_drum.txt /tmp/requirements_drum.txt

RUN sh -c "python3 -m venv ${VIRTUAL_ENV} && \
    . ${VIRTUAL_ENV}/bin/activate && \
    python3 -m ensurepip --default-pip && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade  setuptools wheel pytest && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements_drum.txt && \
    find ${VIRTUAL_ENV} -type d -name '__pycache__' -exec rm -rf {} +"

