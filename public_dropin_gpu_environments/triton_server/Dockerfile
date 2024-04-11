# https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-24-01.html#rel-24-01
FROM nvcr.io/nvidia/tritonserver:24.01-py3

# Install the list of core requirements, e.g. sklearn, numpy, pandas, flask.
# **Don't modify this file!**
COPY dr_requirements.txt dr_requirements.txt

RUN pip3 install -U pip && \
    pip3 install --no-cache-dir -r dr_requirements.txt  && \
    rm -rf dr_requirements.txt

# Copy the drop-in environment code into the correct directory
# Code from the custom model tarball can overwrite the code here
ENV CODE_DIR=/opt/code ADDRESS=0.0.0.0:8080
WORKDIR ${CODE_DIR}
COPY ./*.sh ${CODE_DIR}/

ENV WITH_ERROR_SERVER=1

ENTRYPOINT ["/opt/code/start_server.sh"]
