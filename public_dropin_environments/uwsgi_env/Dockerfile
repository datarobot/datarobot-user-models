# This is the default base image for use with user models and workflows.
FROM datarobot/java-dropin-env-base
#FROM yakovg/java-codegen-nginx

RUN groupadd -g 1000 datarobot && useradd -u 1000 -g 1000 dr
# Copy the drop-in environment code into the correct directory
# Code from the custom model tarball can overwrite the code here

RUN apt-get update && apt-get install -y vim nginx

RUN chmod 707 /var/lib/nginx

COPY mlpiper-2.2.1rc2-py2.py3-none-any.whl mlpiper-2.2.1rc2-py2.py3-none-any.whl

COPY datarobot_drum-1.1.5rc1-py3-none-any.whl datarobot_drum-1.1.5rc1-py3-none-any.whl

RUN pip3 install -U datarobot_drum-1.1.5rc1-py3-none-any.whl
RUN pip3 install -U mlpiper-2.2.1rc2-py2.py3-none-any.whl

COPY ./ /opt/code/
