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
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/dependencies
WORKDIR /tmp/dependencies
RUN git clone https://github.com/pybind/pybind11 && \
    cd pybind11 && \
    git checkout v2.13.6 && \
    mkdir build && \
    cd build && \
    cmake -G "Ninja" .. -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . --config Release&& \
    cmake --install .

RUN git clone https://github.com/TUM-CAMP-NARVIS/xylt.git && \
    cd xylt && \
    mkdir build && \
    cd build && \
    cmake -G "Ninja" .. -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . --config Release && \
    cmake --install . --prefix=/usr

WORKDIR /