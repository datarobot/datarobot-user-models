# This is the default base image for use with user models and workflows.
# It contains a variety of common useful data-science packages and tools.
FROM datarobot/python3-dropin-env-base

# Install the list of core requirements, e.g. sklearn, numpy, pandas, flask.
# **Don't modify this file!**
COPY dr_requirements.txt dr_requirements.txt

# '--upgrade-strategy eager' will upgrade installed dependencies
# according to package requirements or to the latest
RUN pip3 install -U --upgrade-strategy eager --no-cache-dir --prefer-binary -r dr_requirements.txt  && \
    rm -rf dr_requirements.txt

# Install the list of custom Python requirements, e.g. keras, xgboost, etc.
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt --no-cache-dir && \
    rm -rf requirements.txt

# get and install  julia 1.5.4
ENV HOME /opt
RUN apt-get update && apt-get install wget -y && \
    wget https://julialang-s3.julialang.org/bin/linux/x64/1.5/julia-1.5.4-linux-x86_64.tar.gz && \
    tar zxvf julia-1.5.4-linux-x86_64.tar.gz && \
    mkdir $HOME/.julia && \
    mkdir $HOME/julia

ENV JULIA_HOME /julia-1.5.4
ENV JULIA_SYS_IMAGE $HOME/julia/sys.so
ENV JULIA_PROJECT $HOME/julia
ENV PATH $PATH:$JULIA_HOME/bin
COPY sysim.jl $HOME/julia/sysim.jl
COPY . /opt/code
ENV PYTHON /usr/local/bin/python 
RUN julia "$HOME/julia/sysim.jl" && \
    chmod -R 777 $HOME/.julia/logs/manifest_usage.toml && \
    rm julia-1.5.4-linux-x86_64.tar.gz

ENV HOME=/opt CODE_DIR=/opt/code ADDRESS=0.0.0.0:8080
WORKDIR /opt/code

ENV WITH_ERROR_SERVER=1
#Uncomment the following line to switch from Flask to uwsgi server
#ENV PRODUCTION=1 MAX_WORKERS=1 SHOW_STACKTRACE=1

ENTRYPOINT ["/opt/code/start_server.sh"]  
