# This is the default base image for use with user models and workflows.
# It contains a variety of common useful data-science packages and tools.
FROM datarobot/python3-dropin-env-base

RUN apt-get update && apt-get install -y vim

# Install the list of core requirements, e.g. sklearn, numpy, pandas, flask.
# **Don't modify this file!**
#COPY dr_requirements.txt dr_requirements.txt

RUN pip3 install \
    --extra-index-url https://artifactory.int.datarobot.com/artifactory/api/pypi/python-all/simple \
    datarobot-mlops

# Using our copy of drum since it is not pushed yet to pypi
COPY datarobot_drum-1.2.1rc5-py3-none-any.whl /tmp
RUN pip3 install /tmp/datarobot_drum-1.2.1rc5-py3-none-any.whl



# '--upgrade-strategy eager' will upgrade installed dependencies
# according to package requirements or to the latest
#RUN pip3 install --upgrade-strategy eager --no-cache-dir --prefer-binary -r dr_requirements.txt  && \
#    rm -rf dr_requirements.txt

# Install the list of custom Python requirements, e.g. keras, xgboost, etc.
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt --no-cache-dir && \
    rm -rf requirements.txt

# Copy the drop-in environment code into the correct directory
# Code from the custom model tarball can overwrite the code here
COPY ./ /opt/code/
