# syntax = docker/dockerfile:experimental
ARG UBUNTU_VERSION=20.04

ARG ARCH=
ARG CUDA=11.2
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}.1-base-ubuntu${UBUNTU_VERSION} as base
ARG CUDA
ARG CUDNN=8.1.0.77-1
ARG CUDNN_MAJOR_VERSION=8
ARG LIB_DIR_PREFIX=x86_64
ARG LIBNVINFER=8.0.0-1
ARG LIBNVINFER_MAJOR_VERSION=8
# Let us install tzdata painlessly
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Avoid confirmation dialogs
ENV DEBIAN_FRONTEND=noninteractive
# Makes Poetry behave more like npm, with deps installed inside a .venv folder
# See https://python-poetry.org/docs/configuration/#virtualenvsin-project
ENV POETRY_VIRTUALENVS_IN_PROJECT=true

# CUDA drivers
SHELL ["/bin/bash", "-c"]
COPY dependencies/install_cuda.sh ./install_cuda.sh
RUN ./install_cuda.sh && \
    rm install_cuda.sh

# System dependencies
RUN apt update && apt install -y wget git python3 python3-pip zip ffmpeg libsm6 libxext6

# Install cmake for tensorflow-onnx
COPY dependencies/install_cmake.sh install_cmake.sh
RUN /bin/bash install_cmake.sh && \
    rm install_cmake.sh

# Install TensorFlow
COPY dependencies/install_tensorflow.sh install_tensorflow.sh
RUN /bin/bash install_tensorflow.sh && \
    rm install_tensorflow.sh

# Install TensorFlow addons
COPY dependencies/install_tensorflow_addons.sh install_tensorflow_addons.sh
RUN /bin/bash install_tensorflow_addons.sh && \
    rm install_tensorflow_addons.sh

# Install onnx-tensorflow
RUN git clone https://github.com/onnx/onnx-tensorflow.git && \
    cd onnx-tensorflow && \
    git checkout 3f87e6235c96f2f66e523d95dc35ff4802862231 && \
    pip3 install -e .

# Local dependencies
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

COPY requirements-nvidia.txt ./
RUN pip3 install -r requirements-nvidia.txt

# YOLOX dependencies (which are of course not versioned again)
COPY yolox-repo/requirements.txt ./yolox-repo/requirements.txt
RUN pip3 install -r yolox-repo/requirements.txt

# numpy 1.24.0 causes issues with tensorflow 2.6.0. "AttributeError: module 'numpy' has no attribute 'object'"
RUN python3 -m pip install numpy==1.23.4

# Grab weights
RUN wget http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/detection/coco/edgeai-yolox/yolox_nano_ti_lite_26p1_41p8_checkpoint.pth

# Rest of YOLOX
COPY yolox-repo/ ./yolox-repo

WORKDIR /scripts

# Copy the normal files (e.g. run.sh and the extract_dataset scripts in)
COPY . ./

ENTRYPOINT ["/bin/bash", "run.sh"]
