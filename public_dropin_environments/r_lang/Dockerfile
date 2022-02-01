# This is the default base image for use with user models and workflows.
# It contains a variety of common useful data-science packages and tools.
FROM datarobot/r-dropin-env-base

RUN apt update -y && apt install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt update -y && apt install -y python3.7 python3.7-dev

RUN python3.7 -m pip install setuptools wheel virtualenv && \
    python3.7 -m virtualenv /opt/v3.7

ENV PATH="/opt/v3.7/bin:$PATH"

# Install the list of core requirements, e.g. numpy, pandas, flask, rpy2.
# **Don't modify this file!**
COPY dr_requirements.txt dr_requirements.txt

# TODO: remove once pyarrow is pinned to the same version across the repo
RUN python3 -m pip install pip==20.2.4
# '--upgrade-strategy eager' will upgrade dependencies
# according to package requirements or to the latest
RUN python3 -m pip install -U --upgrade-strategy eager -r dr_requirements.txt --no-cache-dir && \
    rm -rf dr_requirements.txt

ENV HOME=/opt CODE_DIR=/opt/code ADDRESS=0.0.0.0:8080
WORKDIR ${CODE_DIR}
COPY ./ ${CODE_DIR}

ENV WITH_ERROR_SERVER=1
# Uncomment the following line to switch from Flask to uwsgi server
#ENV PRODUCTION=1 MAX_WORKERS=1 SHOW_STACKTRACE=1

ENTRYPOINT ["/opt/code/start_server.sh"]
