# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(tcn_zenoh_receiver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(holoscan 3.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(CUDAToolkit REQUIRED)

# zenoh-c / zenoh-cpp (same discovery pattern as tcn_zenoh_subscriber)
find_package(zenohc QUIET)
find_package(zenohcxx QUIET)

if(NOT zenohc_FOUND)
    find_library(ZENOHC_LIB zenohc
                 PATHS "/usr/lib" "/usr/local/lib")
    find_path(ZENOHC_INCLUDE "zenohc.hxx"
              PATHS "/usr/include" "/usr/local/include")
    if(ZENOHC_LIB AND ZENOHC_INCLUDE)
        if(NOT TARGET zenohc::lib)
            add_library(zenohc::lib SHARED IMPORTED)
            set_target_properties(zenohc::lib PROPERTIES
                IMPORTED_LOCATION "${ZENOHC_LIB}"
                INTERFACE_INCLUDE_DIRECTORIES "${ZENOHC_INCLUDE}"
            )
        endif()
        set(zenohc_FOUND TRUE)
    endif()
endif()

if(NOT zenohcxx_FOUND)
    find_path(ZENOHCXX_INCLUDE "zenoh.hxx"
              PATHS "/usr/include" "/usr/local/include")
    if(ZENOHCXX_INCLUDE)
        if(NOT TARGET zenohcxx::zenohcxx)
            add_library(zenohcxx::zenohcxx INTERFACE IMPORTED)
            set_target_properties(zenohcxx::zenohcxx PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${ZENOHCXX_INCLUDE}"
            )
            if(TARGET zenohc::lib)
                set_property(TARGET zenohcxx::zenohcxx APPEND PROPERTY
                    INTERFACE_LINK_LIBRARIES zenohc::lib)
            endif()
        endif()
        set(zenohcxx_FOUND TRUE)
    endif()
endif()

if(NOT zenohc_FOUND OR NOT zenohcxx_FOUND)
    message(WARNING "zenoh-c/zenoh-cpp not found - tcn_zenoh_receiver will not be built.\n"
                    "  Install zenoh-c and zenoh-cpp or set ZENOHC_LIB/ZENOHCXX_INCLUDE.")
    return()
endif()

# tcn_cdr_serde is a sibling operator (provides CdrTypeRegistry)
if(NOT TARGET holoscan::ops::tcn_cdr_serde)
    message(WARNING "tcn_cdr_serde not available - tcn_zenoh_receiver will not be built.")
    return()
endif()

# ---------------------------------------------------------------------------
# Main library
# ---------------------------------------------------------------------------
add_library(tcn_zenoh_receiver SHARED
    zenoh_receiver_op.cpp
)
add_library(holoscan::ops::tcn_zenoh_receiver ALIAS tcn_zenoh_receiver)

target_include_directories(tcn_zenoh_receiver PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_compile_definitions(tcn_zenoh_receiver PRIVATE ZENOHCXX_ZENOHC=1)

target_link_libraries(tcn_zenoh_receiver PUBLIC
    holoscan::core
    holoscan::ops::tcn_cdr_serde
    CUDA::cudart
)
if(TARGET zenohcxx::zenohcxx)
    target_link_libraries(tcn_zenoh_receiver PUBLIC zenohcxx::zenohcxx)
elseif(TARGET zenohc::lib)
    target_link_libraries(tcn_zenoh_receiver PUBLIC zenohc::lib)
endif()

if(HOLOHUB_BUILD_PYTHON)
  add_subdirectory(python)
endif()
