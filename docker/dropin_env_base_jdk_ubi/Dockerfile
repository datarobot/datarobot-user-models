# This is the default base image for use with user models and workflows.
FROM redhat/ubi8-minimal:8.8-1037

ARG DATAROBOT_MLOPS_VERSION

ENV JARS_PATH=/opt/jars/

COPY requirements.txt requirements.txt

RUN microdnf update && \
    microdnf install -y python3.11 python3.11-pip python3.11-devel \
    gcc-8.5.0 gcc-c++-8.5.0 which \
    java-11-openjdk-headless-1:11.0.25.0.9 java-11-openjdk-devel-1:11.0.25.0.9 \
    tar-2:1.30 gzip-1.9 unzip-6.0 zip-3.0 wget-1.19.5 vim-minimal-2:8.0.1763 nano-2.9.8 && \
    # try to squeeze everything into one layer
    pip3 install -U pip --no-cache-dir && \
    pip3 install --no-cache-dir wheel setuptools && \
    pip3 install --no-cache-dir -r requirements.txt && \
    microdnf upgrade && \
    microdnf clean all
COPY datarobot-mlops-*.jar $JARS_PATH
COPY mlops-agent-*.jar $JARS_PATH
ENV DRUM_JAVA_SHARED_JARS=$JARS_PATH/*
ENV MLOPS_MONITORING_AGENT_JAR_PATH=$JARS_PATH/mlops-agent-${DATAROBOT_MLOPS_VERSION}.jar
# This makes print statements show up in the logs API
ENV PYTHONUNBUFFERED=1
