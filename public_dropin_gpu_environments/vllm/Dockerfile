FROM vllm/vllm-openai:v0.6.0
USER root
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
  && rm -rf /var/lib/apt/lists/*

# Don't send any telemetry data (vLLM or HuggingFace libraries)
ENV DO_NOT_TRACK=1
ENV PIP_NO_CACHE_DIR=1
ENV CODE_DIR=/opt/code ADDRESS=0.0.0.0:8080

ENV DATAROBOT_VENV_PATH=/workspace/datarobot
RUN mkdir -p ${DATAROBOT_VENV_PATH} && chown -R 1000:1000 /root ${DATAROBOT_VENV_PATH}

USER 1000
RUN python3 -m venv --system-site-packages --symlinks $DATAROBOT_VENV_PATH && $DATAROBOT_VENV_PATH/bin/pip install -U pip
COPY dr_requirements.txt dr_requirements.txt
RUN $DATAROBOT_VENV_PATH/bin/pip install --no-warn-conflicts -r dr_requirements.txt

# Make sure these cache dirs are writable by the vLLM process
ENV HF_HOME=$CODE_DIR/.cache/huggingface
ENV NUMBA_CACHE_DIR=$CODE_DIR/.cache/numba
ENV OUTLINES_CACHE_DIR=$CODE_DIR/.cache/outlines

WORKDIR ${CODE_DIR}
COPY --chown=1000:1000 ./*.sh ./*.py ${CODE_DIR}/

ENTRYPOINT ["/opt/code/start_server.sh"]
