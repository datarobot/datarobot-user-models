FROM vllm/vllm-openai:v0.7.2
USER root
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-venv \
  && rm -rf /var/lib/apt/lists/*

# Don't send any telemetry data (vLLM or HuggingFace libraries)
ENV DO_NOT_TRACK=1
ENV PIP_NO_CACHE_DIR=1

ENV CODE_DIR=/opt/code ADDRESS=0.0.0.0:8080
ENV DATAROBOT_VENV_PATH=/opt/venv
ENV HOME=/opt/.home

COPY --chown=1000:0 .gitkeep ${HOME}/.placeholder
COPY --chown=1000:0 .gitkeep ${CODE_DIR}/.placeholder
RUN chmod g=u ${CODE_DIR} ${HOME}

USER 1000:0
RUN python3.9 -m venv $DATAROBOT_VENV_PATH && $DATAROBOT_VENV_PATH/bin/pip install -U pip
COPY requirements.txt requirements.txt
RUN $DATAROBOT_VENV_PATH/bin/pip install -r requirements.txt

# A lot of the GenAI libraries write files to homedir so we need to be sure it is set to a
# path that is writeable.
ENV PATH=${DATAROBOT_VENV_PATH}/bin:$PATH

WORKDIR ${CODE_DIR}
COPY --chown=1000:0 ./*.sh ./*.py ${CODE_DIR}/

ENTRYPOINT ["/opt/code/start_server.sh"]
