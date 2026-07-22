# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.20)
project(tcn_zenoh_receiver LANGUAGES NONE)

find_package(holoscan 3.7.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Enable the operators
# NOTE: this convenience target is named with an `_app` suffix to avoid
# target-name collisions in the shared CMake tree when HOLOHUB_BUILD_PYTHON=ON.
# The bare name `tcn_zenoh_receiver` is the operator library
# (operators/tcn_artekmed/tcn_zenoh_receiver) and `tcn_zenoh_receiver_python`
# is that operator's pybind11 module (created by pybind11_add_holohub_module).
# This mirrors the cpp app's distinct `tcn_zenoh_receiver_cpp` target. The name
# is not referenced elsewhere.
add_library(tcn_zenoh_receiver_app INTERFACE)
target_link_libraries(tcn_zenoh_receiver_app INTERFACE
    holoscan::core
    holoscan::ops::format_converter
    holoscan::ops::holoviz
    holoscan::ops::nv_video_decoder
    #holoscan::ops::tcn_stream_synchronizer
)