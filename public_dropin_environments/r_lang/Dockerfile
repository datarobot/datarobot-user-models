# This is the default base image for use with user models and workflows.
# It contains a variety of common useful data-science packages and tools.
FROM datarobot/r-dropin-env-base

# Install the list of core requirements, e.g. numpy, pandas, flask, rpy2.
# **Don't modify this file!**
COPY dr_requirements.txt dr_requirements.txt

# '--upgrade-strategy eager' will upgrade dependencies
# according to package requirements or to the latest
RUN pip3 install --upgrade-strategy eager -r dr_requirements.txt --no-cache-dir && \
    rm -rf dr_requirements.txt

COPY ./ /opt/code/
