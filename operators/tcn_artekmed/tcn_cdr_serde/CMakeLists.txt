# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(tcn_cdr_serde LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(holoscan 3.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(fastcdr REQUIRED)

# tcn_schema provides the IDL-generated message types
# Try CMake config first, then manual discovery
find_package(tcn_schema QUIET CONFIG)
if(NOT tcn_schema_FOUND)
    # Manual discovery: look for installed headers and libraries
    find_path(TCN_SCHEMA_INCLUDE_DIR "tcnart_msgs/msg/VideoStream.h"
              PATHS "/usr/include" "/usr/local/include")
    find_library(TCN_SCHEMA_TCNART_LIB tcn_schema_tcnart_msgs
                 PATHS "/usr/lib" "/usr/local/lib")
    find_library(TCN_SCHEMA_STD_LIB tcn_schema_std_msgs
                 PATHS "/usr/lib" "/usr/local/lib")
    find_library(TCN_SCHEMA_BUILTIN_LIB tcn_schema_builtin_interfaces
                 PATHS "/usr/lib" "/usr/local/lib")
    find_library(TCN_SCHEMA_PCPD_LIB tcn_schema_pcpd_msgs
                 PATHS "/usr/lib" "/usr/local/lib")

    if(TCN_SCHEMA_INCLUDE_DIR AND TCN_SCHEMA_TCNART_LIB)
        message(STATUS "Using tcn_schema from: ${TCN_SCHEMA_INCLUDE_DIR}")
        if(NOT TARGET tcn_schema::tcnart_msgs)
            add_library(tcn_schema::tcnart_msgs STATIC IMPORTED)
            set_target_properties(tcn_schema::tcnart_msgs PROPERTIES
                IMPORTED_LOCATION "${TCN_SCHEMA_TCNART_LIB}"
                INTERFACE_INCLUDE_DIRECTORIES "${TCN_SCHEMA_INCLUDE_DIR}"
            )
        endif()
        if(TCN_SCHEMA_STD_LIB AND NOT TARGET tcn_schema::std_msgs)
            add_library(tcn_schema::std_msgs STATIC IMPORTED)
            set_target_properties(tcn_schema::std_msgs PROPERTIES
                IMPORTED_LOCATION "${TCN_SCHEMA_STD_LIB}"
                INTERFACE_INCLUDE_DIRECTORIES "${TCN_SCHEMA_INCLUDE_DIR}"
            )
        endif()
        if(TCN_SCHEMA_BUILTIN_LIB AND NOT TARGET tcn_schema::builtin_interfaces)
            add_library(tcn_schema::builtin_interfaces STATIC IMPORTED)
            set_target_properties(tcn_schema::builtin_interfaces PROPERTIES
                IMPORTED_LOCATION "${TCN_SCHEMA_BUILTIN_LIB}"
                INTERFACE_INCLUDE_DIRECTORIES "${TCN_SCHEMA_INCLUDE_DIR}"
            )
        endif()
        if(TCN_SCHEMA_PCPD_LIB AND NOT TARGET tcn_schema::pcpd_msgs)
            add_library(tcn_schema::pcpd_msgs STATIC IMPORTED)
            set_target_properties(tcn_schema::pcpd_msgs PROPERTIES
                IMPORTED_LOCATION "${TCN_SCHEMA_PCPD_LIB}"
                INTERFACE_INCLUDE_DIRECTORIES "${TCN_SCHEMA_INCLUDE_DIR}"
            )
        endif()
        set(tcn_schema_FOUND TRUE)
    endif()
endif()

if(NOT tcn_schema_FOUND)
    message(WARNING "tcn_schema not found - tcn_cdr_serde will not be built.\n"
                    "  Install tcn_schema or set TCN_SCHEMA_INCLUDE_DIR.")
    return()
endif()

# Also need the fastrtps fixed_size_string.hpp stub (shipped by tcn_schema)
find_path(FASTRTPS_STUB_INCLUDE "fastrtps/utils/fixed_size_string.hpp"
          PATHS "/usr/include" "/usr/local/include" "${TCN_SCHEMA_INCLUDE_DIR}/..")

# ---------------------------------------------------------------------------
# Main library
# ---------------------------------------------------------------------------
add_library(tcn_cdr_serde SHARED
    cdr_serde.cpp
)
add_library(holoscan::ops::tcn_cdr_serde ALIAS tcn_cdr_serde)

target_include_directories(tcn_cdr_serde PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
if(FASTRTPS_STUB_INCLUDE)
    target_include_directories(tcn_cdr_serde PUBLIC "${FASTRTPS_STUB_INCLUDE}")
endif()

target_link_libraries(tcn_cdr_serde PUBLIC
    holoscan::core
    fastcdr
    tcn_schema::tcnart_msgs
    tcn_schema::std_msgs
    tcn_schema::builtin_interfaces
)
if(TARGET tcn_schema::pcpd_msgs)
    target_link_libraries(tcn_cdr_serde PUBLIC tcn_schema::pcpd_msgs)
endif()
