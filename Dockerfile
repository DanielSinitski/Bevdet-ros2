FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ARG TORCH_VERSION=1.10.0
ARG TORCHVISION_VERSION=0.11.0
ARG ONNXRUNTIME_VERSION=1.17.0
ARG MMCV_VERSION=1.5.3
ARG PPLCV_VERSION=0.7.0
ARG MMDEPLOY_VERSION=1.0.0
ENV FORCE_CUDA="1"
ENV DEBIAN_FRONTEND=noninteractive

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
    locales \
    curl \
    lsb-release \
    gnupg \
    software-properties-common \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-setuptools \
    python-is-python3 \
    build-essential \
    wget \
    git \
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

# ROS 2 Humble
RUN locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    export LANG=en_US.UTF-8 && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/ros2.list && \
    apt update && \
    apt install -y ros-humble-desktop python3-colcon-common-extensions

# Venv + Pip Setup
RUN python3 -m pip install --upgrade pip && \
    python3 -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --upgrade pip setuptools wheel

ENV PATH="/opt/venv/bin:$PATH"

# PyTorch + Torchvision
RUN pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
    --extra-index-url https://download.pytorch.org/whl/cu117 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# mmcv-full
RUN pip install mmcv-full==${MMCV_VERSION} \
    -f https://download.openmmlab.com/mmcv/dist/cu117/torch${TORCH_VERSION}/index.html \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /root/workspace

# ONNX Runtime
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz && \
    tar -zxvf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz && \
    pip install onnxruntime-gpu==${ONNXRUNTIME_VERSION} -i https://pypi.tuna.tsinghua.edu.cn/simple

ENV ONNXRUNTIME_DIR=/root/workspace/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}
ENV TENSORRT_DIR=/workspace/tensorrt

# MMDeploy
RUN git clone https://github.com/open-mmlab/mmdeploy.git && \
    cd mmdeploy && \
    git checkout tags/v${MMDEPLOY_VERSION} -b tag_v${MMDEPLOY_VERSION} && \
    git submodule update --init --recursive && \
    mkdir -p build && cd build && \
    cmake -DMMDEPLOY_TARGET_BACKENDS="ort;trt" .. && \
    make -j$(nproc) && cd .. && \
    pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

# Build ppl.cv
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
    mmdet3d==1.0.0rc4 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
