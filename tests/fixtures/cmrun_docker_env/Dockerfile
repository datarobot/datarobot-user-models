# This is the default base image for use with user models and workflows.
# It contains a variety of common useful data-science packages and tools.
FROM python:3.7-slim
ENV LC_ALL=en_US.UTF-8 TERM=xterm COLS=132 ROWS=43 DEBIAN_FRONTEND=noninteractive

# Install dependencies for python packages that may not be part of their wheels
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        libgomp1 gcc libc6-dev

# Package versions are the latest as of when this script was last updated,
# unless otherwise noted.
RUN pip3 install \
    wheel==0.32.3

# Install the list of custom Python requirements, e.g. keras, xgboost, etc.
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt --no-cache-dir && \
    rm -rf requirements.txt

# Install drum requirements separately to avoid installing them every time drum recompiled
COPY drum_requirements.txt drum_requirements.txt
RUN pip3 install -r drum_requirements.txt --no-cache-dir && \
    rm -rf drum_requirements.txt

# Copying the fresh wheel file
COPY datarobot_drum-*.whl /tmp/
RUN ls /tmp/ && \
    ww=$(find /tmp/datarobot_drum*.whl) && \
    pip3 install -U --no-deps $ww
