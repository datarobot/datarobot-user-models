# This is the default base image for use with user models and workflows.
FROM datarobot/java-dropin-env-base

# Install the list of core requirements, e.g. sklearn, numpy, pandas, flask.
# **Don't modify this file!**
COPY dr_requirements.txt dr_requirements.txt

# '--upgrade-strategy eager' will upgrade installed dependencies
# according to package requirements or to the latest
RUN pip3 install -U --upgrade-strategy eager --no-cache-dir -r dr_requirements.txt --use-deprecated=legacy-resolver && \
    rm -rf dr_requirements.txt

# Copy the drop-in environment code into the correct directory
# Code from the custom model tarball can overwrite the code here
COPY ./ /opt/code/
