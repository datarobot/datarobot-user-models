# This is the default base image for use with user models and workflows.
# It contains a variety of common useful data-science packages and tools.
FROM rocker/tidyverse:4.2.1

RUN echo '22 Aug 2022'

ENV LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
          python3-pip \
          python3-dev \
          libglpk-dev \
          nginx \
          && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    chmod 707 /var/lib/nginx

RUN Rscript -e "install.packages('caret', Ncpus=4)" && \
    Rscript -e "install.packages('recipes', Ncpus=4)" && \
    Rscript -e "install.packages('glmnet', Ncpus=4)" && \
    Rscript -e "install.packages('plumber', Ncpus=4)" && \
    Rscript -e "install.packages('Rook', Ncpus=4)" && \
    Rscript -e "install.packages('rjson', Ncpus=4)" && \
    Rscript -e "install.packages('e1071', Ncpus=4)"

# # Install caret models
RUN Rscript -e 'library(caret); install.packages(unique(modelLookup()[modelLookup()$forReg, c(1)]), Ncpus=4)'
RUN Rscript -e 'library(caret); install.packages(unique(modelLookup()[modelLookup()$forClass, c(1)]), Ncpus=4)'

RUN rm -rf /tmp/downloaded_packages/ /tmp/*.rds

COPY requirements.txt requirements.txt
RUN pip3 install -U pip && pip3 install -r requirements.txt