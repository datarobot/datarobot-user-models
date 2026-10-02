ARG PYTHON_VERSION=3.12

FROM vllm/vllm-openai:v0.11.0
USER root

ARG PYTHON_VERSION
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
  && rm -rf /var/lib/apt/lists/*

# Don't send any telemetry data (vLLM or HuggingFace libraries)
ENV DO_NOT_TRACK=1
ENV PIP_NO_CACHE_DIR=1

ENV CODE_DIR=/opt/code ADDRESS=0.0.0.0:8080
ENV DATAROBOT_VENV_PATH=/opt/venv
ENV HOME=/opt/.home

RUN groupadd -g 1000 datarobot && \
    useradd -u 1000 -g datarobot -m -d $HOME -s /bin/bash datarobot

USER 1000:0

COPY --chown=1000:0 .gitkeep ${HOME}/.placeholder
COPY --chown=1000:0 .gitkeep ${CODE_DIR}/.placeholder
COPY --chown=1000:0 .gitkeep ${DATAROBOT_VENV_PATH}/.placeholder
RUN chmod -R g=u ${CODE_DIR} ${HOME} ${DATAROBOT_VENV_PATH}

ARG PYTHON_VERSION
RUN python${PYTHON_VERSION} -m venv $DATAROBOT_VENV_PATH && ${DATAROBOT_VENV_PATH}/bin/pip install -U pip
ENV PATH=${DATAROBOT_VENV_PATH}/bin:$PATH
# This variable has to be populated as of the current commit's DRUM version, as vllm predictor in DRUM
# uses it to craft a correct PATH to launch vllm from a system python.
ENV VIRTUAL_ENV=${DATAROBOT_VENV_PATH}

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR ${CODE_DIR}
COPY --chown=1000:0 ./*.sh ./*.py ${CODE_DIR}/

ENV WITH_ERROR_SERVER=0
 #This makes print statements show up in the logs API
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/opt/code/start_server.sh"]
