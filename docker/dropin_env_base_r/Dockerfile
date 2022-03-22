# This is the default base image for use with user models and workflows.
# It contains a variety of common useful data-science packages and tools.
FROM ubuntu:18.04

RUN echo '29 Aug 2019'

ENV LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 DEBIAN_FRONTEND=noninteractive

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
        pandoc \
        nginx \
        && \
        apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    chmod 707 /var/lib/nginx

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

RUN pip3 install -U pip
RUN pip3 install --no-cache-dir setuptools wheel

### Save cran as the default repo for R packages
RUN echo "r <- getOption('repos'); r['CRAN'] <- 'http://cran.rstudio.com'; options(repos = r);" > ~/.Rprofile

# Install R Packages
RUN Rscript -e "install.packages('devtools', Ncpus=4)" && \
    Rscript -e "install.packages('tidyverse', Ncpus=4)" && \
    Rscript -e "install.packages('caret', Ncpus=4)" && \
    Rscript -e "install.packages('recipes', Ncpus=4)" && \
    Rscript -e "install.packages('glmnet', Ncpus=4)" && \
    Rscript -e "install.packages('plumber', Ncpus=4)" && \
    Rscript -e "install.packages('Rook', Ncpus=4)" && \
    Rscript -e "install.packages('rjson', Ncpus=4)" && \
    Rscript -e "install.packages('e1071', Ncpus=4)"

# Install caret models
RUN Rscript -e 'library(caret); install.packages(unique(modelLookup()[modelLookup()$forReg, c(1)]), Ncpus=4)'
RUN Rscript -e 'library(caret); install.packages(unique(modelLookup()[modelLookup()$forClass, c(1)]), Ncpus=4)'

RUN rm -rf /tmp/downloaded_packages/ /tmp/*.rds

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt