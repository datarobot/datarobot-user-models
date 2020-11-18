FROM ubuntu:18.04
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
        openjdk-11-jdk \
        openjdk-11-jre \
        libgomp1 \
        gcc \
        libc6-dev \
        pandoc \
        nginx \
        git \
        maven \
        vim \
        docker.io && \
    rm -rf /var/lib/apt/lists/*

RUN echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen && \
    locale-gen en_US.utf8 && \
    /usr/sbin/update-locale LANG=en_US.UTF-8

RUN pip3 install -U pip
RUN pip3 install -U setuptools wheel
COPY requirements_drum.txt /tmp/requirements_drum.txt
RUN pip3 install -r /tmp/requirements_drum.txt && rm -rf /root/.cache/pip
