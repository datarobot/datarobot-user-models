FROM nvcr.io/nim/nvidia/llama-3.2-nv-embedqa-1b-v2:1.3.1

USER root
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    python3-wheel \
    git \
  && rm -rf /var/lib/apt/lists/*

ENV DATAROBOT_VENV_PATH=/opt/drum
ENV PIP_NO_CACHE_DIR=1
RUN /usr/bin/python3 -m venv $DATAROBOT_VENV_PATH && $DATAROBOT_VENV_PATH/bin/pip3 install -U pip
COPY requirements.txt requirements.txt

ENV PIP_INDEX_URL=https://pypi.org/simple
RUN $DATAROBOT_VENV_PATH/bin/pip install -r requirements.txt

ENV CODE_DIR=/opt/code ADDRESS=0.0.0.0:8080
ENV NIM_CACHE_PATH=${CODE_DIR}/nim/cache
WORKDIR ${CODE_DIR}
COPY --chown=1000:1000 .gitkeep ${NIM_CACHE_PATH}/
COPY --chown=1000:1000 ./*.sh ./*.py ${CODE_DIR}/

ENTRYPOINT ["/opt/code/start_server.sh"]
