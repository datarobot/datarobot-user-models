# This is the default base image for use with user models and workflows.
ARG BASE_ROOT_IMAGE=python:3.11.11-slim-bookworm

FROM ${BASE_ROOT_IMAGE}

# Upgrade setuptools to a secure version (CVE-2024-6345)
RUN pip install --no-cache-dir --upgrade pip setuptools>=70.0.0

# This makes print statements show up in the logs API
ENV PYTHONUNBUFFERED=1
