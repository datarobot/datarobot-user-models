FROM datarobot/dropin-env-base:1.0.0-python-3.11.11-slim-bookworm

RUN useradd -u 1000 -m drum

# install genai-perf tool
# see https://docs.nvidia.com/nim/benchmarking/llm/latest/step-by-step.html
#RUN mkdir -p /.cache && chown -R drum:drum /.cache
#RUN apt update && apt install -y --no-install-recommends git
#RUN git config --global credential.helper store

#RUN apt install -y libstdc++6 gcc g++ make cmake clang
#RUN pip install genai-perf==0.0.10

ENV PATH=/home/drum/.local/bin:$PATH
ENV OPENAI_HOST=localhost
ENV OPENAI_PORT=8000
ENV CODE_DIR=/opt/code
ENV ADDRESS=0.0.0.0:8080

WORKDIR ${CODE_DIR}
COPY . ${CODE_DIR}/

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN chown -R drum:drum ${CODE_DIR}

USER drum

ENTRYPOINT ["/opt/code/start_server.sh"]
