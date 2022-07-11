# This is the default base image for use with user models and workflows.
# It contains a variety of common useful data-science packages and tools.
FROM datarobot/dropin-env-base-r:ubuntu20.04-r4.2.1-py3.8-jre11-drum1.9.5-mlops8.1.3
# Install the list of core requirements, e.g. numpy, pandas, flask, rpy2.
# **Don't modify this file!**
COPY dr_requirements.txt dr_requirements.txt

# '--upgrade-strategy eager' upgrades dependencies
# according to the package requirements or to the latest
RUN python3 -m pip install -U --upgrade-strategy eager -r dr_requirements.txt --no-cache-dir && \
    rm -rf dr_requirements.txt

# required by rpy2 https://github.com/rpy2/rpy2/issues/874
ENV LD_LIBRARY_PATH=/usr/local/lib/R/lib:/usr/local/lib:/usr/lib/x86_64-linux-gnu:/usr/lib/jvm/java-11-openjdk-amd64/lib/server

ENV HOME=/opt CODE_DIR=/opt/code ADDRESS=0.0.0.0:8080
WORKDIR ${CODE_DIR}
COPY ./ ${CODE_DIR}

ENV WITH_ERROR_SERVER=1
# Uncomment the following line to switch from Flask to uwsgi server
#ENV PRODUCTION=1 MAX_WORKERS=1 SHOW_STACKTRACE=1

ENTRYPOINT ["/opt/code/start_server.sh"]
