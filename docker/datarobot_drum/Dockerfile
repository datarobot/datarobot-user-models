# This is the default base image for use with user models and workflows.
# It contains a variety of common useful data-science packages and tools.
FROM python:3.7-slim
ENV LC_ALL=en_US.UTF-8 TERM=xterm COLS=132 ROWS=43 DEBIAN_FRONTEND=noninteractive

# Install dependencies for python packages that may not be part of their wheels
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        libgomp1 \
        gcc \
        libc6-dev \
        nginx \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    chmod 707 /var/lib/nginx

# Package versions are the latest as of when this script was last updated,
# unless otherwise noted.
RUN pip3 install -U pip
RUN pip3 install --no-cache-dir setuptools wheel

# do the trick to install drum deps
RUN pip3 install --no-cache-dir --prefer-binary datarobot-drum

ENTRYPOINT ["drum"]