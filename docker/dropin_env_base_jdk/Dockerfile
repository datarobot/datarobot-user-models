# This is the default base image for use with user models and workflows.
FROM openjdk:11.0.16-jdk-slim-bullseye

RUN apt-get update && apt-get upgrade -y && apt-get install -y python3 python3-pip python3-dev python3-distutils fontconfig nginx && \
    pip3 install --upgrade pip && \
    pip3 install --no-cache-dir wheel setuptools && \
    # required for mlpiper
    apt-get install -y python3-setuptools && \
    chmod -R 707 /var/lib/nginx /var/log/nginx

COPY requirements.txt requirements.txt
COPY datarobot-mlops-*.jar /opt/jars/

ENV DRUM_JAVA_SHARED_JARS=/opt/jars/*

RUN pip3 install -U pip && pip3 install -r requirements.txt
