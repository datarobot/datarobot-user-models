# This is the default base image for use with user models and workflows.
FROM datarobot/dropin-env-base-jdk:debian11-py3.9-jdk11.0.16-drum1.13.0-mlops9.2.8

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
# [DEPRECATED] Uncomment the following line to switch to produciton mode
#ENV PRODUCTION=1 MAX_WORKERS=4 SHOW_STACKTRACE=1

ENTRYPOINT ["/opt/code/start_server.sh"]
