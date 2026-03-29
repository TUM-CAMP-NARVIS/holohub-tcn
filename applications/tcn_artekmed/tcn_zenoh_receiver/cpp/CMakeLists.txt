# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(tcn_zenoh_receiver_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(holoscan 3.7.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(tcn_zenoh_receiver_cpp
    tcn_zenoh_receiver.cpp
)
set_target_properties(tcn_zenoh_receiver_cpp PROPERTIES OUTPUT_NAME tcn_zenoh_receiver)

# yaml-cpp (provided by holoscan SDK)
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

target_link_libraries(tcn_zenoh_receiver_cpp
    holoscan::core
    holoscan::ops::tcn_zenoh_receiver
    holoscan::ops::gxf_codelet
    $<$<BOOL:${yaml-cpp_FOUND}>:yaml-cpp::yaml-cpp>
)

# Copy the YAML config file next to the executable
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/tcn_zenoh_receiver.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/tcn_zenoh_receiver.yaml
    COPYONLY
)
