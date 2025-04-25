FROM nvidia/cuda:11.5.1-cudnn8-devel-ubuntu22.04

ARG PYTHON_VERSION=3.8
ARG TORCH_VERSION=1.10.0
ARG TORCHVISION_VERSION=0.11.0
ARG ONNXRUNTIME_VERSION=1.17.0
ARG MMCV_VERSION=1.5.3
ARG PPLCV_VERSION=0.7.0
ENV FORCE_CUDA="1"
ENV DEBIAN_FRONTEND=noninteractive

# Optional: Mirrors aktivieren
ARG USE_SRC_INSIDE=false
RUN if [ ${USE_SRC_INSIDE} == true ] ; then \
    sed -i s/archive.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list && \
    sed -i s/security.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list ; \
    echo "Use aliyun source for installing libs" ; \
else \
    echo "Keep the download source unchanged" ; \
fi

RUN sed -i s:/archive.ubuntu.com:/mirrors.tuna.tsinghua.edu.cn/ubuntu:g /etc/apt/sources.list && \
    apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python3-venv \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python-is-python3 \
    build-essential \
    curl \
    vim \
    git \
    wget \
    lsb-release \
    gnupg2 \
    locales \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libssl-dev \
    libopencv-dev \
    libspdlog-dev --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# ROS2 Humble installieren
RUN locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    export LANG=en_US.UTF-8 && \
    apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository universe && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add - && \
    echo "deb [arch=amd64 signed-by=/etc/apt/trusted.gpg.d/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list && \
    apt-get update && \
    apt-get install -y ros-humble-desktop python3-colcon-common-extensions && \
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Upgrade pip & virtualenv
RUN python3 -m pip install --upgrade pip && \
    python3 -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --upgrade pip setuptools wheel

ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch
RUN pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --extra-index-url https://download.pytorch.org/whl/cu115 -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install mmcv-full
RUN pip install mmcv-full==${MMCV_VERSION} -f https://download.openmmlab.com/mmcv/dist/cu115/torch${TORCH_VERSION}/index.html -i https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /root/workspace

# Install onnxruntime
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz \
    && tar -zxvf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz \
    && pip install onnxruntime-gpu==${ONNXRUNTIME_VERSION} -i https://pypi.tuna.tsinghua.edu.cn/simple

ENV ONNXRUNTIME_DIR=/root/workspace/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}

# Install mmdeploy
ARG VERSION=1.0.0
RUN git clone https://github.com/open-mmlab/mmdeploy.git && \
    cd mmdeploy && \
    if [ -z ${VERSION} ] ; then echo "No MMDeploy version passed in, building on master" ; else git checkout tags/v${VERSION} -b tag_v${VERSION} ; fi && \
    git submodule update --init --recursive && \
    mkdir -p build && cd build && \
    cmake -DMMDEPLOY_TARGET_BACKENDS="ort" .. && \
    make -j$(nproc) && cd .. && \
    pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

# Build ppl.cv
RUN git clone https://github.com/openppl-public/ppl.cv.git && \
    cd ppl.cv && \
    git checkout tags/v${PPLCV_VERSION} -b v${PPLCV_VERSION} && \
    ./build.sh cuda

ENV BACKUP_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real/:$LD_LIBRARY_PATH

# Rebuild mmdeploy SDK ohne TensorRT
RUN cd /root/workspace/mmdeploy && \
    rm -rf build/CM* build/cmake-install.cmake build/Makefile build/csrc && \
    mkdir -p build && cd build && \
    cmake .. \
        -DMMDEPLOY_BUILD_SDK=ON \
        -DMMDEPLOY_BUILD_EXAMPLES=ON \
        -DCMAKE_CXX_COMPILER=g++ \
        -Dpplcv_DIR=/root/workspace/ppl.cv/cuda-build/install/lib/cmake/ppl \
        -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} \
        -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
        -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
        -DMMDEPLOY_TARGET_BACKENDS="ort" \
        -DMMDEPLOY_CODEBASES=all && \
    make -j$(nproc) && make install && \
    export SPDLOG_LEVEL=warn && \
    echo "Built MMDeploy version v${VERSION} for GPU devices successfully!"

ENV LD_LIBRARY_PATH="/root/workspace/mmdeploy/build/lib:${BACKUP_LD_LIBRARY_PATH}"

# Final Python packages
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
    -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install openmim

RUN pip install mmdet3d==1.0.0rc4

RUN pip install matplotlib scipy Pillow imageio

RUN pip install spconv 

RUN pip install setuptools==59.5.0
