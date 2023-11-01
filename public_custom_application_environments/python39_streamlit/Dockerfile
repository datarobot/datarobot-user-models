FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*


RUN pip3 install --no-cache-dir 'streamlit==1.17.0' 'datarobot==3.0.2' 'plotly==5.13.0' 'streamlit-wordcloud==0.1.0' 'kaleido==0.2.1' 'tabulate==0.9.0' 'altair<5'

USER 1000

WORKDIR /opt/code

EXPOSE 8080
