# Python/Java(JRE) drop-in environments base image
Repository name: **datarobot/dropin-env-base**
Dockerfile: https://github.com/datarobot/datarobot-user-models/blob/master/docker/dropin_env_base/Dockerfile

## Description
This image is used as a base for the Python drop in environments.
Based on openjdk:11.0.15-jre-slim-bullseye.
It contains
* Debian 11
* JRE 11
* Python 3.9
* DRUM 1.9.8
* datarobot 2.28.1
* datarobot-mlops 8.1.3

## Guidelines
DataRobot guidelines for publishing images to Docker Hub
https://datarobot.atlassian.net/wiki/spaces/ENG/pages/927858704/Releasing+Public+Docker+Images+to+Docker+Hub
