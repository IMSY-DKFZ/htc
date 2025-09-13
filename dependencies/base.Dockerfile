# Base components which are needed for cluster jobs and testing
FROM nvidia/cuda:12.8.1-base-ubuntu24.04

# Avoid Docker build freeze due to region selection
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin
RUN apt update && apt-get -y install tzdata

# Basic tools
RUN apt update && apt install -y \
     build-essential \
     curl \
     git \
     rsync \
     tree \
     vim \
     libgl1 libglx-mesa0

# Python (via conda for version management and advanced testing, e.g., for public tests)
ENV PATH="/opt/conda/envs/htc/bin:/opt/conda/condabin:${PATH}"

RUN curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
 && bash Miniforge3-Linux-x86_64.sh -b -p "/opt/conda" \
 && conda update -y --all \
 && conda create --yes --name htc python=3.13 \
 && rm -f Miniforge3-Linux-x86_64.sh

# Ignore root warning since we are in a container
ENV PIP_ROOT_USER_ACTION=ignore

# Cache common pretrained models
RUN curl -L https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth --create-dirs -o /root/.cache/torch/hub/checkpoints/efficientnet-b5-b6417697.pth

# Installing all dependencies at once leads to a huge image layer which can lead to problems. Hence, we move out the largest dependency to a separate layer
RUN python -m pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir torch

# Install all requirements separately so that this step can be cached
COPY dependencies/requirements.txt /requirements.txt
COPY dependencies/requirements-extra.txt /requirements-extra.txt
COPY dependencies/requirements-tests.txt /requirements-tests.txt
COPY dependencies/requirements-dev.txt /requirements-dev.txt
RUN pip install --no-cache-dir -r /requirements-dev.txt

# Folders supposed to be mapped during runtime
RUN mkdir /home/results
ENV PATH_HTC_RESULTS=/home/results

# Directly install the htc package in the container since installing it on the cluster with bound volumes does not work properly (random job crashes)
WORKDIR /home/src
