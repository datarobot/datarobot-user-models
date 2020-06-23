FROM rocker/r-apt:bionic
RUN echo '29 Aug 2019'
ENV LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y
RUN apt-get install -y r-cran-recipes
RUN apt-get install -y r-cran-glmnet
RUN apt-get install -y r-cran-rjson
RUN apt-get install -y r-cran-e1071
RUN apt-get install -y r-cran-plumber
RUN apt-get install -y r-cran-caret
RUN apt-get install -y r-cran-tidyverse
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        apt-utils \
        curl \
        gpg-agent \
        software-properties-common \
        dirmngr \
        libssl-dev \
        ca-certificates \
        locales \
        libcurl4-openssl-dev \
        libxml2-dev \
        python3-pip \
        python3-dev \
        libgomp1 \
        gcc \
        libc6-dev \
        pandoc && \
    rm -rf /var/lib/apt/lists/*
RUN echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen && \
    locale-gen en_US.utf8 && \
    /usr/sbin/update-locale LANG=en_US.UTF-8
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 && \
    add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/'
RUN apt-get install -y --no-install-recommends \
        r-cran-littler \
        r-base \
        r-base-dev && \
    rm -rf /var/lib/apt/lists/
RUN pip3 install setuptools wheel

### Save cran as the default repo for R packages
RUN echo "r <- getOption('repos'); r['CRAN'] <- 'http://cran.rstudio.com'; options(repos = r);" > ~/.Rprofile
# Install R Packages
RUN Rscript -e "install.packages('Rook', Ncpus=6)"
RUN Rscript -e "install.packages('tidyverse', Ncpus=6)"
# Install caret models
RUN Rscript -e 'library(caret); install.packages(unique(modelLookup()[modelLookup()$forReg, c(1)]), Ncpus=4)'
RUN Rscript -e 'library(caret); install.packages(unique(modelLookup()[modelLookup()$forClass, c(1)]), Ncpus=4)'
RUN Rscript -e "install.packages('devtools', Ncpus=4)"

RUN rm -rf /tmp/downloaded_packages/ /tmp/*.rds

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        git \
        maven \
        docker.io &&\
    rm -rf /var/lib/apt/lists/*

RUN apt install software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.7 python3.7-dev



RUN python3.7 -m pip install setuptools
RUN python3.7 -m pip install wheel
RUN python3.7 -m pip install virtualenv

RUN python3.7 -m virtualenv /opt/v3.7
COPY requirements_drum.txt /tmp/requirements_drum.txt
RUN cd /opt && . v3.7/bin/activate && pip install -r /tmp/requirements_drum.txt && rm -rf /root/.cache/pip

COPY requirements_dropin.txt /tmp/requirements_dropin.txt
RUN cd /opt && . v3.7/bin/activate && pip install -r /tmp/requirements_dropin.txt && rm -rf /root/.cache/pip


