# This is the default base image for use with user models and workflows.
# It contains a variety of common useful data-science packages and tools.
FROM datarobot/dropin-env-base:1.0.0-python-3.11.11-slim-bookworm

# Install the list of core requirements, e.g. sklearn, numpy, pandas, flask.
# **Don't modify this file!**
COPY requirements.txt requirements.txt

# '--upgrade-strategy eager' will upgrade installed dependencies
# according to package requirements or to the latest
RUN pip3 install -U pip && \
    pip3 install -U --upgrade-strategy eager --no-cache-dir -r requirements.txt  && \
    rm -rf requirements.txt

# MARK: FUNCTIONAL-TEST-ADD-HERE. (This line is used by DRUM functional test automation and can be safely ignored.)

# Copy the drop-in environment code into the correct directory
# Code from the custom model tarball can overwrite the code here
ENV HOME=/opt CODE_DIR=/opt/code ADDRESS=0.0.0.0:8080
WORKDIR ${CODE_DIR}
COPY ./*.sh ${CODE_DIR}/

ENV WITH_ERROR_SERVER=1

ENTRYPOINT ["/opt/code/start_server.sh"]
