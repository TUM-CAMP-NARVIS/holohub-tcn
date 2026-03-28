# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2025 Ulrich Eck
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

############################################################
# Base image
############################################################

ARG BASE_IMAGE
ARG GPU_TYPE

FROM ${BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive

# --------------------------------------------------------------------------
#
# Holohub dev setup
#
RUN mkdir -p /tmp/scripts
COPY holohub /tmp/scripts/
RUN mkdir -p /tmp/scripts/utilities
COPY utilities /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/holohub
RUN /tmp/scripts/holohub setup && rm -rf /var/lib/apt/lists/*

# Enable autocomplete
RUN echo ". /etc/bash_completion.d/holohub_autocomplete" >> /etc/bash.bashrc

# - This variable is consumed by all dependencies below as an environment variable (CMake 3.22+)
# - We use ARG to only set it at docker build time, so it does not affect cmake builds
#   performed at docker run time in case users want to use a different BUILD_TYPE
ARG CMAKE_BUILD_TYPE=Release

# --------------------------------------------------------------------------
#
# tcn_artekmed specific dependencies
#

RUN apt-get update \
    && apt-get install -y \
      libavformat-dev gdb rustc cargo \
      capnproto libcapnp-dev \
      libeigen3-dev \
      libglfw3-dev \
      libxkbcommon-x11-0 \
      libx11-6 \
      libxcb1 \
      libxcb-dri2-0 \
      libxrender1 \
      libxkbcommon0 \
      libglu1-mesa \
      libgtk-3-0 \
      libwayland-client0 \
      libwayland-cursor0 \
      libwayland-egl1 \
      build-essential \
      ccache \
      gdb \
      python3-dbg \
      git-lfs \
      wget \
      unzip \
      cmake \
      libvulkan-dev \
      libxinerama-dev \
      libxcursor-dev \
      xorg-dev \
      libglu1-mesa-dev \
      pkg-config \
      ninja-build \
      libfontconfig1-dev \
      libfreetype-dev \
      libx11-dev \
      libx11-xcb-dev \
      libxcb-cursor-dev \
      libxcb-glx0-dev \
      libxcb-icccm4-dev \
      libxcb-image0-dev \
      libxcb-keysyms1-dev \
      libxcb-randr0-dev \
      libxcb-render-util0-dev \
      libxcb-shape0-dev \
      libxcb-shm0-dev \
      libxcb-sync-dev \
      libxcb-util-dev \
      libxcb-xfixes0-dev \
      libxcb-xkb-dev \
      libxcb1-dev \
      libxext-dev \
      libxfixes-dev \
      libxi-dev \
      libxkbcommon-dev \
      libxkbcommon-x11-dev \
      libxrender-dev \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Copy over app-specific requirements
COPY applications/tcn_artekmed/requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# custom external libraries
RUN mkdir -p /tmp/dependencies

WORKDIR /tmp/dependencies
RUN wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-graphics/2025_5_0/linux/NVIDIA_Nsight_Graphics_2025.5.0.25335.run && \
    chmod +x NVIDIA_Nsight_Graphics_2025.5.0.25335.run && \
    sh ./NVIDIA_Nsight_Graphics_2025.5.0.25335.run --accept --quiet && \
    rm NVIDIA_Nsight_Graphics_2025.5.0.25335.run

WORKDIR /tmp/dependencies
RUN rm -Rf /tmp/depencencies/pybind11 && \
    git clone https://github.com/pybind/pybind11 && \
    cd pybind11 && \
    git checkout v2.13.6 && \
    mkdir build && \
    cd build && \
    cmake -G "Ninja" .. -DCMAKE_BUILD_TYPE=Release -DPYBIND11_TEST=OFF && \
    cmake --build . --config Release -- -j 8  && \
    cmake --install .

# Upgrade Rust toolchain for iceoryx2 (base image has Cargo 1.75 which is too old)
# Also install libclang-dev for bindgen
RUN apt-get update && apt-get install -y --no-install-recommends libclang-dev && \
    rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable && \
    . "$HOME/.cargo/env" && rustc --version && cargo --version
ENV PATH="/root/.cargo/bin:${PATH}"

# Build and install iceoryx2 v0.8.1 C/C++ bindings (requires Rust toolchain)
WORKDIR /tmp/dependencies
RUN rm -Rf /tmp/dependencies/iceoryx2 && \
    git clone --depth 1 --branch v0.8.1 https://github.com/eclipse-iceoryx/iceoryx2.git && \
    cd iceoryx2 && \
    cargo generate-lockfile && \
    mkdir build && \
    cd build && \
    cmake -G "Ninja" .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF && \
    cmake --build . --config Release -- -j $(nproc) && \
    cmake --install . && \
    cd /tmp/dependencies && \
    rm -rf iceoryx2

# Build and install eProsima Fast-CDR 2.0.0 (CDR serialization for Zenoh)
WORKDIR /tmp/dependencies
RUN rm -rf /tmp/dependencies/Fast-CDR && \
    git clone --depth 1 --branch v2.0.0 https://github.com/eProsima/Fast-CDR.git && \
    cd Fast-CDR && \
    mkdir build && \
    cd build && \
    cmake -G "Ninja" .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr -DBUILD_TESTING=OFF && \
    cmake --build . --config Release -- -j $(nproc) && \
    cmake --install . && \
    cd /tmp/dependencies && \
    rm -rf Fast-CDR

# Build and install zenoh-c + zenoh-cpp 1.3.4 (Zenoh C++ API)
WORKDIR /tmp/dependencies
RUN rm -rf /tmp/dependencies/zenoh-c && \
    git clone --depth 1 --branch 1.3.4 https://github.com/eclipse-zenoh/zenoh-c.git && \
    cd zenoh-c && \
    mkdir build && \
    cd build && \
    cmake -G "Ninja" .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr -DZENOHC_BUILD_TESTS_WITH_CXX=OFF && \
    cmake --build . --config Release -- -j $(nproc) && \
    cmake --install . && \
    cd /tmp/dependencies && \
    rm -rf zenoh-c

WORKDIR /tmp/dependencies
RUN rm -rf /tmp/dependencies/zenoh-cpp && \
    git clone --depth 1 --branch 1.3.4 https://github.com/eclipse-zenoh/zenoh-cpp.git && \
    cd zenoh-cpp && \
    mkdir build && \
    cd build && \
    cmake -G "Ninja" .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr -DZENOHCXX_ZENOHC=ON && \
    cmake --build . --config Release -- -j $(nproc) && \
    cmake --install . && \
    cd /tmp/dependencies && \
    rm -rf zenoh-cpp

# Build and install tcn_schema (IDL-generated CDR message types)
# Note: upstream CMakeLists.txt unconditionally requires fastddsgen.jar even
# with -DWITH_DDS=OFF.  We patch lines 23-26 (the include + find_file) to
# skip them when DDS is disabled, avoiding the Java/fastddsgen dependency.
WORKDIR /tmp/dependencies
RUN rm -rf /tmp/dependencies/tcn_schema && \
    git clone --depth 1 https://github.com/TUM-CAMP-NARVIS/tcn_schema.git && \
    cd tcn_schema && \
    sed -i '22,26s/^/#/' CMakeLists.txt && \
    mkdir build && \
    cd build && \
    cmake -G "Ninja" .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr \
        -DWITH_DDS=OFF -DBUILD_SHARED_LIBS=OFF && \
    cmake --build . --config Release -- -j $(nproc) && \
    cmake --install . && \
    cd /tmp/dependencies && \
    rm -rf tcn_schema

WORKDIR /tmp/dependencies
RUN rm -Rf /tmp/dependencies/xylt && \
    git clone https://github.com/TUM-CAMP-NARVIS/xylt.git && \
    cd xylt && \
    mkdir build && \
    cd build && \
    cmake -G "Ninja" .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
    cmake --build . --config Release && \
    cmake --install . --prefix=/usr && \
    cp /tmp/dependencies/xylt/build/pyxylt.cpython*.so /usr/local/lib/python3.12/dist-packages/

# Download and install latest Slang release (adjust version as needed)
WORKDIR /tmp/dependencies/slang
ARG SLANG_VERSION=2025.24.2  # Check https://github.com/shader-slang/slang/releases
RUN wget https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}-linux-x86_64.zip \
    && unzip slang-${SLANG_VERSION}-linux-x86_64.zip -d /tmp/dependencies/slang \
    && rm slang-${SLANG_VERSION}-linux-x86_64.zip \
    && ln -s /tmp/dependencies/slang/bin/slangc /usr/local/bin/slangc \
    && ln -s /tmp/dependencies/slang/bin/slangd /usr/local/bin/slangd

# Set PATH and verify
ENV PATH="/usr/local/bin:${PATH}"

WORKDIR /

## Dev container
FROM base as dev


ARG WORKSPACE_DIR=/workspace/holohub

# Keep the user as holoscan since the Holoscan Container already has the user setup
ARG USERNAME=holoscan
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Remove existing conflicting user
# see https://bugs.launchpad.net/cloud-images/+bug/2005129
RUN getent passwd $USER_UID | cut -d: -f1 | xargs -r userdel -r || true

# Remove existing conflicting home directory
RUN if [ -d "/home/$USERNAME" ] ; then ls -l /home/$USERNAME && rm -rf /home/$USERNAME ; fi

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && chown -R $USERNAME /home/$USERNAME

# [Optional] Uncomment this section to install additional vcpkg ports.
# RUN su vscode -c "${VCPKG_ROOT}/vcpkg install <your-port-name-here>"

# Set the default user.
# (See https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user)
USER $USERNAME

ENV PYTHONPATH=/usr/local/lib/python3.12/dist-packages:$PYTHONPATH