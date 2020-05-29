# This is the default base image for use with user models and workflows.
FROM ubuntu:18.04
ENV LC_ALL=en_US.UTF-8 TERM=xterm COLS=132 ROWS=43 DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y locales && locale-gen $LC_ALL && \
    apt-get install --no-install-recommends -y \
            python3-pip \
            python3-dev \
            openjdk-11-jdk \
            openjdk-11-jre \
            gcc \
            libc6-dev \
            && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install setuptools wheel
RUN pip3 install datarobot-drum==1.1.0 --no-cache-dir

# Copy the drop-in environment code into the correct directory
# Code from the custom model tarball can overwrite the code here
COPY ./ /opt/code/
