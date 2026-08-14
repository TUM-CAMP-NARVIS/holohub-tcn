#!/bin/bash

# --------------------------------------------------------------------------
#
# tcn_artekmed specific dependencies
#

apt-get update
apt-get install -y \
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
      libxrender-dev 

# Install Python dependencies
python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

cd /tmp/dependencies
wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-graphics/2025_5_0/linux/NVIDIA_Nsight_Graphics_2025.5.0.25335.run && \
    chmod +x NVIDIA_Nsight_Graphics_2025.5.0.25335.run && \
    sh ./NVIDIA_Nsight_Graphics_2025.5.0.25335.run --accept --quiet && \
    rm NVIDIA_Nsight_Graphics_2025.5.0.25335.run

cd /tmp/dependencies
rm -Rf /tmp/dependencies/pybind11 && \
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
apt-get update && apt-get install -y --no-install-recommends libclang-dev && \
    rm -rf /var/lib/apt/lists/*
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable && \
    . "$HOME/.cargo/env" && rustc --version && cargo --version
export PATH="/root/.cargo/bin:${PATH}"

# Build and install iceoryx2 v0.9.2 C/C++ bindings (requires Rust toolchain)
cd /tmp/dependencies
rm -Rf /tmp/dependencies/iceoryx2 && \
    git clone --depth 1 --branch v0.9.2 https://github.com/eclipse-iceoryx/iceoryx2.git && \
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
cd /tmp/dependencies
rm -rf /tmp/dependencies/Fast-CDR && \
    git clone --depth 1 --branch v2.0.0 https://github.com/eProsima/Fast-CDR.git && \
    cd Fast-CDR && \
    mkdir build && \
    cd build && \
    cmake -G "Ninja" .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr -DBUILD_TESTING=OFF && \
    cmake --build . --config Release -- -j $(nproc) && \
    cmake --install . && \
    cd /tmp/dependencies && \
    rm -rf Fast-CDR

# Build and install zenoh-c + zenoh-cpp 1.9.0 (Zenoh C++ API)
cd /tmp/dependencies
rm -rf /tmp/dependencies/zenoh-c && \
    git clone --depth 1 --branch 1.9.0 https://github.com/eclipse-zenoh/zenoh-c.git && \
    cd zenoh-c && \
    mkdir build && \
    cd build && \
    cmake -G "Ninja" .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr -DZENOHC_BUILD_TESTS_WITH_CXX=OFF && \
    cmake --build . --config Release -- -j $(nproc) && \
    cmake --install . && \
    cd /tmp/dependencies && \
    rm -rf zenoh-c

cd /tmp/dependencies
rm -rf /tmp/dependencies/zenoh-cpp && \
    git clone --depth 1 --branch 1.9.0 https://github.com/eclipse-zenoh/zenoh-cpp.git && \
    cd zenoh-cpp && \
    mkdir build && \
    cd build && \
    cmake -G "Ninja" .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr -DZENOHCXX_ZENOHC=ON && \
    cmake --build . --config Release -- -j $(nproc) && \
    cmake --install . && \
    cd /tmp/dependencies && \
    rm -rf zenoh-cpp

# Install Java runtime + build fastddsgen (IDL-to-C++ code generator for tcn_schema)
# Java 17 specifically — Gradle 7.6 (used by fastddsgen v3.x) doesn't support Java 21
apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jdk-headless

cd /tmp/dependencies
rm -rf /tmp/dependencies/Fast-DDS-Gen && \
    git clone --depth 1 --branch v3.0.0 --recurse-submodules \
        https://github.com/eProsima/Fast-DDS-Gen.git && \
    cd Fast-DDS-Gen && \
    ./gradlew assemble && \
    mkdir -p /usr/share/fastddsgen/java && \
    cp share/fastddsgen/java/fastddsgen.jar /usr/share/fastddsgen/java/ && \
    cd /tmp/dependencies && \
    rm -rf Fast-DDS-Gen

# Build and install tcn_schema (IDL-generated CDR message types)
cd /tmp/dependencies
rm -rf /tmp/dependencies/tcn_schema && \
    git clone --depth 1 https://github.com/TUM-CAMP-NARVIS/tcn_schema.git && \
    cd tcn_schema && \
    mkdir build && \
    cd build && \
    cmake -G "Ninja" .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr \
        -DWITH_DDS=OFF -DBUILD_SHARED_LIBS=OFF \
        -DFASTDDS_GEN_JAR_PATH=/usr/share/fastddsgen/java \
        -DCMAKE_INSTALL_INCLUDEDIR=include && \
    cmake --build . --config Release -- -j $(nproc) && \
    cmake --install . && \
    cd /tmp/dependencies && \
    rm -rf tcn_schema

# @todo: hardcodes python version version!!
cd /tmp/dependencies
rm -Rf /tmp/dependencies/xylt && \
    git clone https://github.com/TUM-CAMP-NARVIS/xylt.git && \
    cd xylt && \
    mkdir build && \
    cd build && \
    cmake -G "Ninja" .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
    cmake --build . --config Release && \
    cmake --install . --prefix=/usr && \
    cp /tmp/dependencies/xylt/build/pyxylt.cpython*.so /usr/local/lib/python3.12/dist-packages/

# Download and install latest Slang release (adjust version as needed)
# @todo: hardcodes platform and slang version!!
mkdir -p /tmp/dependencies/slang && cd /tmp/dependencies/slang
export SLANG_VERSION=2026.14.1  # Check https://github.com/shader-slang/slang/releases
wget https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}-linux-x86_64.zip \
    && unzip -o slang-${SLANG_VERSION}-linux-x86_64.zip -d /tmp/dependencies/slang \
    && rm slang-${SLANG_VERSION}-linux-x86_64.zip \
    && ln -sf /tmp/dependencies/slang/bin/slangc /usr/local/bin/slangc \
    && ln -sf /tmp/dependencies/slang/bin/slangd /usr/local/bin/slangd


# Cleanup
apt-get autoremove -y
apt-get clean -y
rm -rf /var/lib/apt/lists/*

