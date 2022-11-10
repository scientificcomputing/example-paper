FROM ubuntu:22.04

ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system

# Install pip and git with apt
RUN apt-get update && \
    apt-get install -y python3-pip git

# We upgrade pip and setuptools
RUN python3 -m pip install --no-cache-dir pip setuptools --upgrade

WORKDIR /tmp

# Copy pyproject.toml first so that we done need to reinstall in case anoter file
# is changing ater rebuiding docker image
COPY pyproject.toml  .
RUN python3 -m pip install pip --upgrade && python3 -m pip install --no-cache-dir . && rm -rf /tmp

WORKDIR /app
COPY . /app
