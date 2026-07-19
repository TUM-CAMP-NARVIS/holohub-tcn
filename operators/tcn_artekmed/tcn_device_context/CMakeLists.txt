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
project(tcn_device_context LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(holoscan 3.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Eigen3 is required by xylt's cmake config
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    find_path(EIGEN3_INCLUDE_DIR Eigen/Core
              PATHS /usr/include/eigen3 /usr/local/include/eigen3)
    if(EIGEN3_INCLUDE_DIR AND NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
    endif()
endif()

# Require xylt library for XY lookup table generation
find_package(xylt QUIET)

# Fix xylt's cmake config: it may have a bogus "/include" in INTERFACE_INCLUDE_DIRECTORIES
# (happens when CMAKE_INSTALL_PREFIX was "/" during xylt's build). Replace with actual path.
if(xylt_FOUND AND TARGET xylt)
    get_target_property(_xylt_incs xylt INTERFACE_INCLUDE_DIRECTORIES)
    if(_xylt_incs)
        list(REMOVE_ITEM _xylt_incs "/include")
        find_path(_xylt_real_inc xylt/library.h PATHS /usr/include /usr/local/include)
        if(_xylt_real_inc)
            list(APPEND _xylt_incs "${_xylt_real_inc}")
        endif()
        list(REMOVE_DUPLICATES _xylt_incs)
        set_target_properties(xylt PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_xylt_incs}")
    endif()
endif()

if(NOT xylt_FOUND)
    # Manual discovery: check common install locations
    find_library(XYLT_LIBRARY xylt
                 PATHS /usr/lib /usr/local/lib
                 NO_DEFAULT_PATH)
    find_path(XYLT_INCLUDE_DIR xylt/library.h
              PATHS /usr/include /usr/local/include
              NO_DEFAULT_PATH)

    if(XYLT_LIBRARY AND XYLT_INCLUDE_DIR)
        message(STATUS "Using xylt from: ${XYLT_LIBRARY}")
        if(NOT TARGET xylt::xylt)
            add_library(xylt::xylt SHARED IMPORTED)
            set_target_properties(xylt::xylt PROPERTIES
                IMPORTED_LOCATION "${XYLT_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${XYLT_INCLUDE_DIR}"
            )
        endif()
        set(xylt_FOUND TRUE)
    endif()
endif()

if(NOT xylt_FOUND)
    message(WARNING "xylt not found - tcn_device_context will not be built.\n"
                    "  Install xylt from https://github.com/TUM-CAMP-NARVIS/xylt.git")
    return()
endif()

# Determine xylt target name
if(TARGET xylt::xylt)
    set(XYLT_TARGET xylt::xylt)
elseif(TARGET xylt)
    set(XYLT_TARGET xylt)
else()
    message(FATAL_ERROR "xylt found but no usable target")
endif()

# ---------------------------------------------------------------------------
# Main library
# ---------------------------------------------------------------------------
add_library(tcn_device_context SHARED
    device_context_service.cpp
    xy_lookup_table_source_op.cpp
)
add_library(holoscan::ops::tcn_device_context ALIAS tcn_device_context)

target_include_directories(tcn_device_context PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(tcn_device_context PUBLIC
    holoscan::core
    holoscan::ops::tcn_shm_serde
    ${XYLT_TARGET}
    cuda
)

if(HOLOHUB_BUILD_PYTHON)
  add_subdirectory(python)
endif()
