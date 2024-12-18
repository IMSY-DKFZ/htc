FROM nvidia/cuda:12.4.1-base-ubuntu22.04

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
     libgl1-mesa-glx libglib2.0-0

# Python
ENV PATH="/opt/conda/bin:${PATH}"

RUN curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
 && bash Miniforge3-Linux-x86_64.sh -b -p "/opt/conda" \
 && rm -f Miniforge3-Linux-x86_64.sh

# Cache common pretrained models
RUN curl -L https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth --create-dirs -o /root/.cache/torch/hub/checkpoints/efficientnet-b5-b6417697.pth

# Install all requirements separately so that this step can be cached
COPY requirements.txt /requirements.txt
COPY requirements-extra.txt /requirements-extra.txt
COPY requirements-dev.txt /requirements-dev.txt
RUN python -m pip install -U pip \
 && pip install -r /requirements-dev.txt

# Folders supposed to be mapped during runtime
RUN mkdir /home/results
ENV PATH_HTC_RESULTS=/home/results

# Directly install the htc package in the container since installing it on the cluster with bound volumes does not work properly (random job crashes)
WORKDIR /home/src
