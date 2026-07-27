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
project(tcn_shm_backprojection_test LANGUAGES NONE)

find_package(holoscan 3.7.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Enable the operators
add_library(tcn_shm_backprojection_test INTERFACE)
target_link_libraries(tcn_shm_backprojection_test INTERFACE
    holoscan::core
    holoscan::ops::format_converter
    holoscan::ops::holoviz

    holoscan::ops::tcn_shm_subscriber
    holoscan::ops::tcn_device_context
    holoscan::ops::tcn_stream_splitter
    holoscan::ops::tcn_stream_merger
    holoscan::ops::tcn_flatten_tensor
    holoscan::ops::tcn_depthimage_backprojection
    holoscan::ops::tcn_depthimage_temporal_filter
)