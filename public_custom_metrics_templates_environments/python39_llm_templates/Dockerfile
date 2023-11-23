# This is the default base image for use with user models and workflows.
# It contains a variety of common useful data-science packages and tools.
FROM datarobot/dropin-env-base:debian11-py3.9-jre11.0.16-drum1.10.7-mlops9.1.3

# Install the list of custom Python requirements, e.g. pandas, numpy, etc.
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt --no-cache-dir && \
    rm -rf requirements.txt

# Download the nltk packages required to run nltk models
RUN [ "python3", "-c", "import nltk; nltk.download('popular', download_dir='/opt/nltk_data')"]

# Copy the drop-in environment code into the correct directory
ENV HOME=/opt CODE_DIR=/opt/code ADDRESS=0.0.0.0:8080
WORKDIR ${CODE_DIR}
COPY ./*.sh ${CODE_DIR}/
