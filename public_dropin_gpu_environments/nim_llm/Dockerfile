FROM nvcr.io/nim/meta/llama-3.1-8b-instruct:1.2.2

USER root
RUN apt-get update && apt-get install -y \
    curl \
    python3-pip \
    python3-venv \
    python3-wheel \
  && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_linux_amd64.deb -o s5cmd.deb \
  && dpkg -i s5cmd.deb \
  && rm s5cmd.deb

ENV DATAROBOT_VENV_PATH=/opt/drum
ENV PIP_NO_CACHE_DIR=1
RUN /usr/bin/python3 -m venv $DATAROBOT_VENV_PATH && $DATAROBOT_VENV_PATH/bin/pip3 install -U pip
COPY requirements.txt requirements.txt

RUN $DATAROBOT_VENV_PATH/bin/pip install -r requirements.txt

ENV CODE_DIR=/opt/code ADDRESS=0.0.0.0:8080
ENV NIM_CACHE_PATH=${CODE_DIR}/nim/cache
WORKDIR ${CODE_DIR}
COPY --chown=1000:1000 .gitkeep ${NIM_CACHE_PATH}/
COPY --chown=1000:1000 ./*.sh ./*.py ${CODE_DIR}/

ENTRYPOINT ["/opt/code/start_server.sh"]
