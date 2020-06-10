# This is the default base image for use with user models and workflows.
FROM datarobot/java-dropin-env-base

# '--upgrade-strategy eager' will upgrade installed dependencies
# according to package requirements or to the latest
RUN pip3 install --upgrade-strategy eager --no-cache-dir datarobot-drum==1.1.0

# Copy the drop-in environment code into the correct directory
# Code from the custom model tarball can overwrite the code here
COPY ./ /opt/code/
