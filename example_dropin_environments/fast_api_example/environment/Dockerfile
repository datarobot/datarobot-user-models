FROM python:3.11-slim

# Set working directory
WORKDIR /opt/code/

# Copy files into the image
COPY . /opt/code/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
