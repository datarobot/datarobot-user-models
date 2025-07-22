FROM datarobotdev/mirror_chainguard_datarobot.com_python-fips:3.12-dev

# This makes print statements show up in the logs API
ENV PYTHONUNBUFFERED=1

# This allows code to access ~ and not default to /, which may not be accessible
ENV HOME=/opt/code

# Add .local/bin to PATH for user-installed packages (gunicorn, flask, etc.)
ENV PATH=/opt/code/.local/bin:$PATH

WORKDIR /opt/code

EXPOSE 8080
