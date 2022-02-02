# This is the default base image for use with user models and workflows.
FROM openjdk:11.0.12-jre-slim-buster


RUN apt update && apt install -y python3 python3-pip python3-distutils fontconfig nginx && \
    pip3 install --upgrade pip && \
    pip3 install --no-cache-dir wheel setuptools && \
    # required for mlpiper
    apt install -y python3-setuptools && \
    chmod 707 /var/lib/nginx

# do the trick to install drum deps
RUN pip3 install --no-cache-dir datarobot-drum && pip3 uninstall -y datarobot-drum
