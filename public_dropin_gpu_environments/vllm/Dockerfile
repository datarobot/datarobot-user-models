FROM vllm/vllm-openai:v0.7.2
USER root
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
  && rm -rf /var/lib/apt/lists/*

# Don't send any telemetry data (vLLM or HuggingFace libraries)
ENV DO_NOT_TRACK=1
ENV PIP_NO_CACHE_DIR=1

ENV CODE_DIR=/opt/code ADDRESS=0.0.0.0:8080
ENV DATAROBOT_VENV_PATH=/opt/venv
ENV HOME=/opt/.home

COPY --chown=1000:0 .gitkeep ${HOME}/.placeholder
COPY --chown=1000:0 .gitkeep ${CODE_DIR}/.placeholder
COPY --chown=1000:0 .gitkeep ${DATAROBOT_VENV_PATH}/.placeholder
RUN chmod g=u ${CODE_DIR} ${HOME} ${DATAROBOT_VENV_PATH}

USER 1000:0
RUN python3.12 -m venv $DATAROBOT_VENV_PATH && ${DATAROBOT_VENV_PATH}/bin/pip install -U pip
ENV PATH=${DATAROBOT_VENV_PATH}/bin:$PATH

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR ${CODE_DIR}
COPY --chown=1000:0 ./*.sh ./*.py ${CODE_DIR}/

ENV WITH_ERROR_SERVER=1

ENTRYPOINT ["/opt/code/start_server.sh"]
