# This is a private Chainguard development image that is stored in DataRobot's private registry.
# Replace it with your own development Chainguard image if you build your own.
FROM datarobotdev/mirror_chainguard_datarobot.com_python-fips:3.11-dev

USER root

# hadolint ignore=DL3018
RUN apk add --no-cache uv

# This makes print statements show up in the logs API
ENV PYTHONUNBUFFERED=1

# This allows code to access ~ and not default to /, which may not be accessible
ENV HOME=/opt/code

# Add .local/bin to PATH for user-installed packages
ENV PATH=/opt/code/.local/bin:$PATH

# uv configuration: use system Python, no auto-downloads, compile bytecode
ENV UV_NO_MANAGED_PYTHON=1
ENV UV_PYTHON_DOWNLOADS=never
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /opt/code

EXPOSE 8080
