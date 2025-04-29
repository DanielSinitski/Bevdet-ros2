# This is the ROS2 Base Image for DNN deployment in the STADT:up Consortia with cuda installed

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ENV FORCE_CUDA="1"
ENV DEBIAN_FRONTEND=noninteractive

USER root
RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update
RUN apt update && apt upgrade -y && apt install --no-install-recommends -y \
    git \
    build-essential \
    curl \
    software-properties-common \
    locales

RUN locale-gen en_US en_US.UTF-8
RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
RUN LANG=en_US.UTF-8

RUN add-apt-repository universe
RUN apt -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update && apt upgrade -y
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update && apt upgrade -y
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt install -y ros-humble-desktop

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    git \
    sudo \
    python3-pip \
    python-is-python3 \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    python3-rosdep \
    python3-vcstool \
    && rm -rf /var/lib/apt/lists/*

# bootstrap rosdep
RUN rosdep init && \
  rosdep update --rosdistro humble

# setup colcon mixin and metadata
RUN colcon mixin add default \
      https://raw.githubusercontent.com/colcon/colcon-mixin-repository/master/index.yaml && \
    colcon mixin update && \
    colcon metadata add default \
      https://raw.githubusercontent.com/colcon/colcon-metadata-repository/master/index.yaml && \
    colcon metadata update

# install ros2 packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ros-base=0.10.0-1* \
    && rm -rf /var/lib/apt/lists/*

# install basic ros2 packages
RUN apt update && apt upgrade -y && apt install --no-install-recommends -y \
    ros-humble-sensor-msgs \
    ros-humble-std-msgs \
    ros-humble-cv-bridge \
    ros-humble-vision-msgs \
    ros-humble-rmw-cyclonedds-cpp

# Install basic python packages
RUN pip3 install \
    opencv-python \
    cv_bridge \
    numpy

# change shell
SHELL ["/bin/bash", "--login", "-c"]

# Mirror einstellen (optional)
ARG USE_SRC_INSIDE=false
RUN if [ "${USE_SRC_INSIDE}" = "true" ]; then \
    sed -i s/archive.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list && \
    sed -i s/security.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list ; \
    echo "Use aliyun source for installing libs" ; \
else \
    echo "Keep the download source unchanged" ; \
fi

# Python 3.10 + ROS 2 Humble + Grundpakete
RUN sed -i s:/archive.ubuntu.com:/mirrors.tuna.tsinghua.edu.cn/ubuntu:g /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y \
    lsb-release \
    gnupg \
    python3 \
    python3-venv \
    python3-dev \
    python3-setuptools \
    wget \
    vim \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libssl-dev \
    libopencv-dev \
    libspdlog-dev \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Venv + Pip Setup
RUN python3 -m pip install --upgrade pip && \
    python3 -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --upgrade pip setuptools wheel

ENV PATH="/opt/venv/bin:$PATH"

# PyTorch + Torchvision

# PyTorch + Torchvision: Richtiges CUDA Matching
ARG TORCH_VERSION=1.12.1
ARG TORCHVISION_VERSION=0.13.1
RUN pip install torch==${TORCH_VERSION}+cu116 torchvision==${TORCHVISION_VERSION}+cu116 \
    -f https://download.pytorch.org/whl/cu116/torch_stable.html \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# mmcv-full: passt zu torch 1.12.1 + CUDA 11.6/11.7
ARG MMCV_VERSION=1.5.3
ENV TORCH_CUDA_ARCH_LIST="8.6"
RUN pip install mmcv-full==${MMCV_VERSION} \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch1.11/index.html \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /root/workspace

# ONNX Runtime
ARG ONNXRUNTIME_VERSION=1.17.0
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz && \
    tar -zxvf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz && \
    pip install onnxruntime-gpu==${ONNXRUNTIME_VERSION} -i https://pypi.tuna.tsinghua.edu.cn/simple

ENV ONNXRUNTIME_DIR=/root/workspace/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}
ENV TENSORRT_DIR=/workspace/tensorrt

# Install TensorRT
RUN apt-get update && apt-get install -y wget gnupg && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y tensorrt libnvinfer-dev libnvinfer-plugin-dev \
        python3-libnvinfer uff-converter-tf && \
    rm -f cuda-keyring_1.1-1_all.deb

    
ARG MMDEPLOY_VERSION=1.0.0
RUN git clone https://github.com/open-mmlab/mmdeploy.git && \
    cd mmdeploy && \
    git checkout tags/v${MMDEPLOY_VERSION} -b tag_v${MMDEPLOY_VERSION} && \
    git submodule update --init --recursive && \
    mkdir -p build && cd build && \
    cmake -DMMDEPLOY_TARGET_BACKENDS="ort;trt" .. && \
    make -j$(nproc) && cd .. && \
    pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

# Build ppl.cv
ARG PPLCV_VERSION=0.7.0
RUN git clone https://github.com/openppl-public/ppl.cv.git && \
    cd ppl.cv && \
    git checkout tags/v${PPLCV_VERSION} -b v${PPLCV_VERSION} && \
    ./build.sh cuda

ENV BACKUP_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real/:$LD_LIBRARY_PATH

# MMDeploy SDK erneut bauen
RUN cd /root/workspace/mmdeploy && \
    rm -rf build && \
    mkdir -p build && cd build && \
    cmake .. \
        -DMMDEPLOY_BUILD_SDK=ON \
        -DMMDEPLOY_BUILD_EXAMPLES=ON \
        -DCMAKE_CXX_COMPILER=g++ \
        -Dpplcv_DIR=/root/workspace/ppl.cv/cuda-build/install/lib/cmake/ppl \
        -DTENSORRT_DIR=${TENSORRT_DIR} \
        -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} \
        -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
        -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
        -DMMDEPLOY_TARGET_BACKENDS="ort;trt" \
        -DMMDEPLOY_CODEBASES=all && \
    make -j$(nproc) && make install && \
    export SPDLOG_LEVEL=warn

ENV LD_LIBRARY_PATH="/root/workspace/mmdeploy/build/lib:${BACKUP_LD_LIBRARY_PATH}"

# Weitere Python Pakete
RUN pip install \
    mmdet==2.25.1 \
    mmsegmentation==0.25.0 \
    pycuda \
    lyft_dataset_sdk \
    networkx==2.2 \
    numba==0.57.1 \
    numpy==1.23.5 \
    nuscenes-devkit \
    plyfile \
    scikit-image \
    tensorboard \
    trimesh==2.35.39 \
    matplotlib \
    scipy \
    Pillow \
    imageio \
    spconv \
    setuptools==59.5.0 \
    openmim \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
