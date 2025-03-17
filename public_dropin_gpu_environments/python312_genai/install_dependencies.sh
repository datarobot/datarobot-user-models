#!/bin/bash
set -e

ARTIFACTORY_MAVEN_URL="https://artifactory.devinfra.drdev.io/artifactory/maven-central/com/datarobot"

# A number of packages here are based on the following custom models image:
# datarobot/dropin-env-base-jdk:ubi8.8-py3.11-jdk11.0.22-drum1.10.20-mlops9.2.8
# (https://github.com/datarobot/datarobot-user-models/blob/master/docker/dropin_env_base_jdk_ubi)
# Downloading MLOps jars prior to build is done via Maven, see pom.xml in the dropin image
# if you need to reproduce the process

# TODO: review dependencies https://datarobot.atlassian.net/browse/BUZZOK-24542
microdnf update
microdnf install -y gcc gcc-c++ which \
  java-11-openjdk-headless-1:11.0.25.0.9 java-11-openjdk-devel-1:11.0.25.0.9 \
  nginx \
  tar gzip unzip zip wget vim-minimal nano

chmod -R 707 /var/lib/nginx /var/log/nginx

pip3 install -U pip --no-cache-dir
pip3 install --no-cache-dir wheel setuptools

pip3 install -r requirements.txt \
  --no-cache-dir \
  --upgrade-strategy eager \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  --extra-index-url https://artifactory.devinfra.drdev.io/artifactory/api/pypi/datarobot-python-dev/simple

mkdir -p $JARS_PATH
curl -L ${ARTIFACTORY_MAVEN_URL}/datarobot-mlops/${DATAROBOT_MLOPS_VERSION}/datarobot-mlops-${DATAROBOT_MLOPS_VERSION}.jar --output ${JARS_PATH}/datarobot-mlops-${DATAROBOT_MLOPS_VERSION}.jar && \
curl -L ${ARTIFACTORY_MAVEN_URL}mlops-agent/${DATAROBOT_MLOPS_VERSION}/mlops-agent-${DATAROBOT_MLOPS_VERSION}.jar --output ${JARS_PATH}/mlops-agent-${DATAROBOT_MLOPS_VERSION}.jar && \

microdnf upgrade
microdnf clean all

rm -rf dep.constraints
rm -rf requirements.txt
