FROM nvcr.io/ohlfw0olaadg/ea-participants/nemollm-inference-ms:dev-24.02


USER root
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    zstd \
  && rm -rf /var/lib/apt/lists/*

USER 1000
ENV MODEL_STORE_PATH=/model-store
ENV DATAROBOT_VENV_PATH=/home/nemo/dr
ENV PIP_NO_CACHE_DIR=1
ENV PATH=${BASE_PATH}
RUN /usr/bin/python3 -m venv $DATAROBOT_VENV_PATH && $DATAROBOT_VENV_PATH/bin/pip3 install -U pip
COPY dr_requirements.txt dr_requirements.txt

RUN $DATAROBOT_VENV_PATH/bin/pip install -r dr_requirements.txt

ENV CODE_DIR=/opt/code ADDRESS=0.0.0.0:8080
WORKDIR ${CODE_DIR}
COPY --chown=1000:1000 ./*.sh ./*.py ${CODE_DIR}/


ENTRYPOINT ["/opt/code/start_server.sh"]
