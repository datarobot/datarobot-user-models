# R drop-in environment base image
Repository name: **datarobot/dropin-env-base-r**  
Latest date: 05-17-2023
Dockerfile: https://github.com/datarobot/datarobot-user-models/blob/master/docker/dropin-env-base-r/Dockerfile

## Description
This image is used as a base for R drop in environments.
Based on Ubuntu:18.04 image. Additionally installed: Python3, R, R libraries, packages required by the **datarobot-drum** tool.

Contains
* Debian 11
* JDK 11
* Python 3.9
* DRUM 1.10.3
* datarobot-mlops 9.0.7

## Guidelines
DataRobot guidelines for publishing images to Docker Hub
https://datarobot.atlassian.net/wiki/spaces/ENG/pages/927858704/Releasing+Public+Docker+Images+to+Docker+Hub
