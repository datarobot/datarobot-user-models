# This is the default base image for use with user models and workflows.
# It contains a variety of common useful data-science packages and tools.

# pypmml uses py4j and java as backend, so use java based env
FROM datarobot/dropin-env-base:debian11-py3.9-jre11.0.16-drum1.13.0-mlops9.2.8

COPY requirements.txt requirements.txt

# '--upgrade-strategy eager' will upgrade installed dependencies
# according to package requirements or to the latest
RUN pip3 install -U pip && \
    pip3 install -U --upgrade-strategy eager --no-cache-dir -r requirements.txt  && \
    rm -rf requirements.txt

# Copy the drop-in environment code into the correct directory
# Code from the custom model tarball can overwrite the code here
ENV HOME=/opt CODE_DIR=/opt/code ADDRESS=0.0.0.0:8080
WORKDIR ${CODE_DIR}
COPY ./*.sh ${CODE_DIR}/

ENV WITH_ERROR_SERVER=1

ENTRYPOINT ["/opt/code/start_server.sh"]
