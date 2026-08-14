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
COPY utilities /tmp/scripts/utilities/
# Copy over app-specific requirements
COPY applications/tcn_artekmed/requirements.txt /tmp/requirements.txt
COPY applications/tcn_artekmed/install_dependencies.sh /tmp/install.sh

# - This variable is consumed by all dependencies below as an environment variable (CMake 3.22+)
# - We use ARG to only set it at docker build time, so it does not affect cmake builds
#   performed at docker run time in case users want to use a different BUILD_TYPE
ARG CMAKE_BUILD_TYPE=Release


RUN mkdir -p /tmp/scripts/utilities && \
    chmod +x /tmp/scripts/holohub && \
    /tmp/scripts/holohub setup && rm -rf /var/lib/apt/lists/* && \
    echo ". /etc/bash_completion.d/holohub_autocomplete" >> /etc/bash.bashrc && \
    mkdir -p /tmp/dependencies && \
    chmod +x /tmp/install.sh && \
    /tmp/install.sh


# Set PATH and verify
ENV PATH="/usr/local/bin:/root/.cargo/bin:${PATH}"
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
