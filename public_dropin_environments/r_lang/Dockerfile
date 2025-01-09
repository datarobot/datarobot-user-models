# This is the default base image for use with user models and workflows.
# It contains a variety of common useful data-science packages and tools.
FROM datarobot/dropin-env-base-r:ubuntu22.04-r4.2.2-py3.10-jre11
# Install the list of core requirements, e.g. numpy, pandas, flask, rpy2.
# **Don't modify this file!**
COPY requirements.txt requirements.txt

# '--upgrade-strategy eager' upgrades dependencies
# according to the package requirements or to the latest
RUN pip3 install -U pip && \
    pip3 install -U --upgrade-strategy eager --no-cache-dir -r requirements.txt  && \
    rm -rf requirements.txt

# required by rpy2 https://github.com/rpy2/rpy2/issues/874
ENV LD_LIBRARY_PATH=/usr/local/lib/R/lib:/usr/local/lib:/usr/lib/x86_64-linux-gnu:/usr/lib/jvm/java-11-openjdk-amd64/lib/server

ENV HOME=/opt CODE_DIR=/opt/code ADDRESS=0.0.0.0:8080
WORKDIR ${CODE_DIR}
COPY ./*.sh ${CODE_DIR}/

ENV WITH_ERROR_SERVER=1

ENTRYPOINT ["/opt/code/start_server.sh"]
