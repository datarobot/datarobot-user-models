# This is the default base image for use with user models and workflows.
FROM openjdk:11.0.16-jre-slim-bullseye

ARG DATAROBOT_MLOPS_VERSION

RUN apt update && apt upgrade -y && apt install -y python3 python3-pip python3-dev python3-distutils fontconfig nginx && \
    pip3 install --upgrade pip && \
    pip3 install --no-cache-dir wheel setuptools && \
    # required for mlpiper
    apt install -y python3-setuptools && \
    chmod -R 707 /var/lib/nginx /var/log/nginx

COPY requirements.txt requirements.txt
COPY datarobot-mlops-*.jar /opt/jars/
COPY mlops-agent-*.jar /opt/jars/

ENV DRUM_JAVA_SHARED_JARS=/opt/jars/*
ENV MLOPS_MONITORING_AGENT_JAR_PATH=/opt/jars/mlops-agent-${DATAROBOT_MLOPS_VERSION}.jar

RUN pip3 install -U pip && pip3 install -r requirements.txt
