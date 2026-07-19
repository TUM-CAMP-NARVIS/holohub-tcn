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

cmake_minimum_required(VERSION 3.24)
project(tcn_shm_receiver_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(holoscan 3.7.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(tcn_shm_receiver_cpp
    tcn_shm_receiver.cpp
)
set_target_properties(tcn_shm_receiver_cpp PROPERTIES OUTPUT_NAME tcn_shm_receiver)

# yaml-cpp is provided by holoscan SDK; find it via pkg-config or system paths
find_package(yaml-cpp QUIET)
if(NOT yaml-cpp_FOUND)
    find_library(YAML_CPP_LIB yaml-cpp)
    find_path(YAML_CPP_INCLUDE yaml-cpp/yaml.h)
    if(YAML_CPP_LIB AND YAML_CPP_INCLUDE)
        add_library(yaml-cpp::yaml-cpp SHARED IMPORTED)
        set_target_properties(yaml-cpp::yaml-cpp PROPERTIES
            IMPORTED_LOCATION "${YAML_CPP_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${YAML_CPP_INCLUDE}")
        set(yaml-cpp_FOUND TRUE)
    endif()
endif()

target_link_libraries(tcn_shm_receiver_cpp
    holoscan::core
    holoscan::ops::holoviz
    holoscan::ops::tcn_shm_subscriber
    holoscan::ops::tcn_device_context
    holoscan::ops::tcn_stream_splitter
    holoscan::ops::tcn_stream_merger
    holoscan::ops::tcn_flatten_tensor
    holoscan::ops::tcn_depthimage_backprojection
    holoscan::ops::tcn_depthimage_temporal_filter
    holoscan::ops::tcn_depthimage_weights
    holoscan::ops::tcn_texture_sampler
    holoscan::ops::tcn_depthimage_max_distance
    holoscan::ops::tcn_depthimage_fgbg_mask
    holoscan::ops::tcn_depthimage_apply_mask
    $<$<BOOL:${yaml-cpp_FOUND}>:yaml-cpp::yaml-cpp>
)

# Copy the YAML config file next to the executable
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/../python/tcn_shm_receiver.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/tcn_shm_receiver.yaml
    COPYONLY
)
