# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(tcn_cdr_decoder LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(holoscan 3.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# ---------------------------------------------------------------------------
# Main library
# ---------------------------------------------------------------------------
add_library(tcn_cdr_decoder SHARED
    cdr_decoder_op.cpp
)
add_library(holoscan::ops::tcn_cdr_decoder ALIAS tcn_cdr_decoder)

target_include_directories(tcn_cdr_decoder PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(tcn_cdr_decoder PUBLIC
    holoscan::core
    holoscan::ops::tcn_cdr_serde
)

if(HOLOHUB_BUILD_PYTHON)
  add_subdirectory(python)
endif()
