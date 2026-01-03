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

# Copy over app-specific requirements
COPY applications/tcn_artekmed/requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

RUN apt-get update \
    && apt-get install -y \
      libavformat-dev gdb rustc cargo \
      libeigen3-dev \
      libxkbcommon-x11-0 \
      libx11-6 \
      libxcb1 \
      libxrender1 \
      libxkbcommon0 \
      libglu1-mesa \
      libgtk-3-0 \
      libwayland-client0 \
      libwayland-cursor0 \
      libwayland-egl1 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/dependencies

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

WORKDIR /tmp/dependencies
RUN rm -Rf /tmp/dependencies/xylt && \
    git clone https://github.com/TUM-CAMP-NARVIS/xylt.git && \
    cd xylt && \
    mkdir build && \
    cd build && \
    cmake -G "Ninja" .. -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . --config Release && \
    cmake --install . --prefix=/usr && \
    cp /tmp/dependencies/xylt/build/pyxylt.cpython*.so /usr/local/lib/python3.12/dist-packages/


# Install additional packages
RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends  \
        build-essential \
        ccache \
        gdb \
        git-lfs \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies for Slang (CMake, Vulkan, etc.)
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    cmake \
    libvulkan-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and install latest Slang release (adjust version as needed)
WORKDIR /tmp/dependencies/slang
ARG SLANG_VERSION=2025.24.2  # Check https://github.com/shader-slang/slang/releases
RUN wget https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}-linux-x86_64.zip \
    && unzip slang-${SLANG_VERSION}-linux-x86_64.zip -d /tmp/dependencies/slang \
    && rm slang-${SLANG_VERSION}-linux-x86_64.zip \
    && ln -s /tmp/dependencies/slang/bin/slangc /usr/local/bin/slangc \
    && ln -s /tmp/dependencies/slang/bin/slangd /usr/local/bin/slangd


# Set PATH and verify
ENV PATH="/tmp/dependencies/slang/bin:${PATH}"

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

ENV PYTHONPATH=/usr/local/lib/python3.10/dist-packages:$PYTHONPATH