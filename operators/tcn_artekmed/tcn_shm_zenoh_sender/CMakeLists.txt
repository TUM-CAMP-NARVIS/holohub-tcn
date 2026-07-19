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
project(tcn_shm_zenoh_sender LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(holoscan 3.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Require iceoryx2 — find all sub-packages for system installs
find_package(iceoryx2-cmake-modules QUIET)
find_package(iceoryx2-c QUIET)
find_package(iceoryx2-bb-cxx QUIET)
find_package(iceoryx2-cxx QUIET)

# System install creates iceoryx2-cxx::shared-lib-cxx, not iceoryx2-cxx::iceoryx2-cxx
if(iceoryx2-cxx_FOUND AND TARGET iceoryx2-cxx::shared-lib-cxx AND NOT TARGET iceoryx2-cxx::iceoryx2-cxx)
    add_library(iceoryx2-cxx::iceoryx2-cxx ALIAS iceoryx2-cxx::shared-lib-cxx)
endif()

if(NOT iceoryx2-cxx_FOUND)
    # Manual discovery from Conan cache (same pattern as tcn_shm_serde)
    if(NOT ICEORYX2_PREFIX)
        file(GLOB _iox_src "$ENV{HOME}/.conan2/p/iceor*/s/src")
        file(GLOB _iox_lib "$ENV{HOME}/.conan2/p/b/iceor*/b/src/target/ff/cc/build/Release/iceoryx2-cxx")
        list(LENGTH _iox_src _iox_src_count)
        list(LENGTH _iox_lib _iox_lib_count)
        if(_iox_src_count GREATER 0 AND _iox_lib_count GREATER 0)
            list(GET _iox_src 0 _iox_src_dir)
            list(GET _iox_lib 0 _iox_lib_dir)
            set(ICEORYX2_PREFIX "${_iox_src_dir}" CACHE PATH "iceoryx2 source root")
            set(ICEORYX2_LIB_DIR "${_iox_lib_dir}" CACHE PATH "iceoryx2 library directory")
        endif()
    endif()

    if(ICEORYX2_PREFIX AND EXISTS "${ICEORYX2_PREFIX}/iceoryx2-cxx/include/iox2/iceoryx2.hpp")
        find_library(ICEORYX2_CXX_LIBRARY iceoryx2_cxx
                     PATHS "${ICEORYX2_LIB_DIR}" NO_DEFAULT_PATH)
        if(ICEORYX2_CXX_LIBRARY)
            message(STATUS "Using iceoryx2 from: ${ICEORYX2_PREFIX}")
            if(NOT TARGET iceoryx2-cxx::iceoryx2-cxx)
                add_library(iceoryx2-cxx::iceoryx2-cxx SHARED IMPORTED)
                set_target_properties(iceoryx2-cxx::iceoryx2-cxx PROPERTIES
                    IMPORTED_LOCATION "${ICEORYX2_CXX_LIBRARY}"
                    INTERFACE_INCLUDE_DIRECTORIES
                        "${ICEORYX2_PREFIX}/iceoryx2-cxx/include;${ICEORYX2_PREFIX}/iceoryx2-bb/cxx/include"
                )
            endif()
            set(iceoryx2-cxx_FOUND TRUE)
        endif()
    endif()
endif()

if(NOT iceoryx2-cxx_FOUND)
    message(WARNING "iceoryx2 not found - tcn_shm_zenoh_sender will not be built.\n"
                    "  Set ICEORYX2_PREFIX to the iceoryx2 source root.")
    return()
endif()

# ---------------------------------------------------------------------------
# Main library
# ---------------------------------------------------------------------------
add_library(tcn_shm_zenoh_sender SHARED
    shm_sender_op.cpp
)
add_library(holoscan::ops::tcn_shm_zenoh_sender ALIAS tcn_shm_zenoh_sender)

target_include_directories(tcn_shm_zenoh_sender PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(tcn_shm_zenoh_sender PUBLIC
    holoscan::core
    holoscan::ops::tcn_shm_serde
    iceoryx2-cxx::iceoryx2-cxx
    cuda
)

if(HOLOHUB_BUILD_PYTHON)
  add_subdirectory(python)
endif()
