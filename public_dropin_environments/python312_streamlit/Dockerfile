FROM python:3.12-slim

WORKDIR /app


RUN pip3 install --no-cache-dir 'streamlit==1.36.0' 'pillow==10.3.0' 'datarobot==3.4.0' 'plotly==5.23.0' 'kaleido==0.2.1' 'tabulate==0.9.0' 'altair<5'

WORKDIR /opt/code

EXPOSE 8080
