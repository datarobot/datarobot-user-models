# This is the default base image for use with user models and workflows.
FROM datarobot/java-dropin-env-base

RUN apt-get update && apt-get install -y vim nginx

RUN chmod 707 /var/lib/nginx

# Install the list of core requirements, e.g. sklearn, numpy, pandas, flask.
# **Don't modify this file!**
COPY dr_requirements.txt dr_requirements.txt

# '--upgrade-strategy eager' will upgrade installed dependencies
# according to package requirements or to the latest
RUN pip3 install --upgrade-strategy eager --no-cache-dir -r dr_requirements.txt  && \
    rm -rf dr_requirements.txt

# Install the list of custom Python requirements, e.g. keras, xgboost, etc.
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt --no-cache-dir && \
    rm -rf requirements.txt

# have to downgrade because of https://github.com/bartdag/py4j/issues/406
RUN pip3 install -U py4j==0.10.7

# Copy the drop-in environment code into the correct directory
# Code from the custom model tarball can overwrite the code here
COPY ./ /opt/code/

RUN chmod 777 /opt/code
