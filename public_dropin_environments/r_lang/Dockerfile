# This is the default base image for use with user models and workflows.
# It contains a variety of common useful data-science packages and tools.
FROM datarobot/dropin-env-base-r:latest

# Install the list of core requirements, e.g. numpy, pandas, flask, rpy2.
# **Don't modify this file!**
COPY dr_requirements.txt dr_requirements.txt

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
