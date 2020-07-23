# This is the default base image for use with user models and workflows.
FROM datarobot/java-dropin-env-base

# Install the list of core requirements, e.g. sklearn, numpy, pandas, flask.
# **Don't modify this file!**
COPY dr_requirements.txt dr_requirements.txt

# '--upgrade-strategy eager' will upgrade installed dependencies
# according to package requirements or to the latest
RUN pip3 install --upgrade-strategy eager --no-cache-dir -r dr_requirements.txt  && \
    rm -rf dr_requirements.txt

# Copy the drop-in environment code into the correct directory
# Code from the custom model tarball can overwrite the code here
COPY ./ /opt/code/

RUN mkdir /opt/code_dep && \ 
    curl https://repo1.maven.org/maven2/ai/h2o/h2o-genmodel/3.30.0.6/h2o-genmodel-3.30.0.6.jar --output /output/code_dep/h2o.jar && \
    h2o_pojo=$(ls /opt/code/ | grep .java) && \
    java -cp /opt/code_dep/h2o.jar /opt/code/$h2o_pojo
