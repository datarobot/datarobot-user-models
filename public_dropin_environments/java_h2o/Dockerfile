# This is the default base image for use with user models and workflows.
FROM datarobot/java-dropin-env-base

COPY dr_requirements.txt dr_requirements.txt

# '--upgrade-strategy eager' will upgrade installed dependencies
# according to package requirements or to the latest
RUN pip3 install --upgrade-strategy eager --no-cache-dir -r dr_requirements.txt  && \
    rm -rf dr_requirements.txt

COPY ./ /opt/code/

RUN apt-get update -y && \
   apt-get install git build-essential maven apt-utils curl gpg-agent software-properties-common dirmngr libssl-dev ca-certificates locales libcurl4-openssl-dev libxml2-dev libgomp1 gcc libc6-dev pandoc -y && \
   git clone -b timsetsfire/h2o_wip_v2 --single-branch https://github.com/datarobot/datarobot-user-models && \
   cd ./datarobot-user-models/custom_model_runner && \ 
   make && \
   cd ../.. && \
   pip3 install -U ./datarobot-user-models/custom_model_runner/dist/datarobot_drum-*.whl --no-deps && \
   apt-get clean && \
   rm -rf /var/lib/apt/lists/*