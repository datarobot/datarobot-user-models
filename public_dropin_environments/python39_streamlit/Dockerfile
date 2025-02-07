FROM python:3.9-slim

# This makes print statements show up in the logs API
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY ./packages/dr_components-0.1.4-py3-none-any.whl /app/

RUN pip3 install -U pip && \
    pip3 install -U --upgrade-strategy eager --no-cache-dir -r requirements.txt  && \
    rm -rf requirements.txt \
RUN pip3 install --no-cache-dir ./dr_components-0.1.4-py3-none-any.whl

WORKDIR /opt/code

EXPOSE 8080
