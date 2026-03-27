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
project(tcn_shm_serde LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Cap'n Proto dependency
# ---------------------------------------------------------------------------
# Try manual discovery first: system CapnProtoConfig.cmake may fail its
# libatomic check even when libatomic1 is installed (Ubuntu 24.04 bug).
# If manual discovery succeeds, we skip find_package entirely.
if(NOT CapnProto_FOUND)
    # Manual discovery: user can set CAPNP_PREFIX to the capnproto install root
    # (e.g. ~/.conan2/p/b/capnpXXX/p)
    if(NOT CAPNP_PREFIX)
        # Auto-detect from Conan cache (pick the first match)
        file(GLOB _capnp_candidates "$ENV{HOME}/.conan2/p/b/capnp*/p")
        list(LENGTH _capnp_candidates _capnp_count)
        if(_capnp_count GREATER 0)
            list(GET _capnp_candidates 0 CAPNP_PREFIX)
        endif()
    endif()

    if(CAPNP_PREFIX AND EXISTS "${CAPNP_PREFIX}/include/capnp/message.h")
        message(STATUS "Using Cap'n Proto from: ${CAPNP_PREFIX}")

        # Find libraries
        find_library(CAPNP_LIB capnp PATHS "${CAPNP_PREFIX}/lib" NO_DEFAULT_PATH)
        find_library(KJ_LIB kj PATHS "${CAPNP_PREFIX}/lib" NO_DEFAULT_PATH)

        if(CAPNP_LIB AND KJ_LIB)
            # Create imported targets to mimic find_package behavior
            if(NOT TARGET CapnProto::kj)
                add_library(CapnProto::kj UNKNOWN IMPORTED)
                set_target_properties(CapnProto::kj PROPERTIES
                    IMPORTED_LOCATION "${KJ_LIB}"
                    INTERFACE_INCLUDE_DIRECTORIES "${CAPNP_PREFIX}/include"
                )
            endif()
            if(NOT TARGET CapnProto::capnp)
                add_library(CapnProto::capnp UNKNOWN IMPORTED)
                set_target_properties(CapnProto::capnp PROPERTIES
                    IMPORTED_LOCATION "${CAPNP_LIB}"
                    INTERFACE_INCLUDE_DIRECTORIES "${CAPNP_PREFIX}/include"
                    INTERFACE_LINK_LIBRARIES "CapnProto::kj"
                )
            endif()
            set(CapnProto_FOUND TRUE)
        endif()
    endif()
endif()

# Fallback: system-installed capnproto (e.g. via apt libcapnp-dev).
# Skip find_package(CapnProto) because its config may fail with libatomic
# check errors on Ubuntu 24.04. Do manual library + include discovery instead.
if(NOT CapnProto_FOUND)
    find_library(CAPNP_LIB capnp)
    find_library(KJ_LIB kj)
    find_path(CAPNP_INCLUDE_DIR capnp/message.h)

    if(CAPNP_LIB AND KJ_LIB AND CAPNP_INCLUDE_DIR)
        message(STATUS "Using system Cap'n Proto: ${CAPNP_LIB}")
        if(NOT TARGET CapnProto::kj)
            add_library(CapnProto::kj UNKNOWN IMPORTED)
            set_target_properties(CapnProto::kj PROPERTIES
                IMPORTED_LOCATION "${KJ_LIB}"
                INTERFACE_INCLUDE_DIRECTORIES "${CAPNP_INCLUDE_DIR}"
            )
        endif()
        if(NOT TARGET CapnProto::capnp)
            add_library(CapnProto::capnp UNKNOWN IMPORTED)
            set_target_properties(CapnProto::capnp PROPERTIES
                IMPORTED_LOCATION "${CAPNP_LIB}"
                INTERFACE_INCLUDE_DIRECTORIES "${CAPNP_INCLUDE_DIR}"
                INTERFACE_LINK_LIBRARIES "CapnProto::kj"
            )
        endif()
        set(CapnProto_FOUND TRUE)
    endif()
endif()

if(NOT CapnProto_FOUND)
    message(WARNING "CapnProto not found - tcn_shm_serde will not be built.\n"
                    "  Options: set CMAKE_PREFIX_PATH, CapnProto_DIR, or CAPNP_PREFIX "
                    "to the capnproto install prefix.")
    return()
endif()

# ---------------------------------------------------------------------------
# Cap'n Proto generated schema sources (pre-compiled from .capnp files)
# ---------------------------------------------------------------------------
set(CAPNP_GENERATED_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/generated/math.capnp.c++
    ${CMAKE_CURRENT_SOURCE_DIR}/generated/enumerations.capnp.c++
    ${CMAKE_CURRENT_SOURCE_DIR}/generated/shm_synchronized_transport.capnp.c++
    ${CMAKE_CURRENT_SOURCE_DIR}/generated/shm_parameter_rpc.capnp.c++
)

# ---------------------------------------------------------------------------
# Core library: tcn_shm_serde (schema types + decode functions)
# ---------------------------------------------------------------------------
add_library(tcn_shm_serde SHARED
    ${CAPNP_GENERATED_SOURCES}
)
add_library(holoscan::ops::tcn_shm_serde ALIAS tcn_shm_serde)

target_include_directories(tcn_shm_serde PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(tcn_shm_serde PUBLIC
    CapnProto::capnp
)

# ---------------------------------------------------------------------------
# Optional: iceoryx2 RPC module
# ---------------------------------------------------------------------------
option(TCN_SHM_SERDE_WITH_RPC "Build RPC module (requires iceoryx2)" ON)

if(TCN_SHM_SERDE_WITH_RPC)
    # Try find_package first — need all iceoryx2 sub-packages in order
    find_package(iceoryx2-cmake-modules QUIET)
    find_package(iceoryx2-c QUIET)
    find_package(iceoryx2-bb-cxx QUIET)
    find_package(iceoryx2-cxx QUIET)

    # System install creates iceoryx2-cxx::shared-lib-cxx, not iceoryx2-cxx::iceoryx2-cxx
    if(iceoryx2-cxx_FOUND AND TARGET iceoryx2-cxx::shared-lib-cxx AND NOT TARGET iceoryx2-cxx::iceoryx2-cxx)
        add_library(iceoryx2-cxx::iceoryx2-cxx ALIAS iceoryx2-cxx::shared-lib-cxx)
    endif()

    if(NOT iceoryx2-cxx_FOUND)
        # Manual discovery from Conan cache
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

    if(iceoryx2-cxx_FOUND)
        message(STATUS "iceoryx2 found - building RPC module")
        target_sources(tcn_shm_serde PRIVATE shm_rpc.cpp)
        target_compile_definitions(tcn_shm_serde PUBLIC TCN_SHM_SERDE_HAS_RPC=1)
        target_link_libraries(tcn_shm_serde PUBLIC iceoryx2-cxx::iceoryx2-cxx)
    else()
        message(STATUS "iceoryx2 not found - RPC module disabled.")
        set(TCN_SHM_SERDE_WITH_RPC OFF)
    endif()
endif()

# ---------------------------------------------------------------------------
# Re-generate Cap'n Proto C++ files (optional convenience target)
# ---------------------------------------------------------------------------
if(CAPNP_PREFIX)
    find_program(CAPNPC_EXECUTABLE capnpc PATHS "${CAPNP_PREFIX}/bin" NO_DEFAULT_PATH)
else()
    find_program(CAPNPC_EXECUTABLE capnpc)
endif()

if(CAPNPC_EXECUTABLE)
    set(SCHEMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tcn_shm_io/schema")
    set(CAPNP_INCLUDE_FLAG "")
    if(CAPNP_PREFIX)
        set(CAPNP_INCLUDE_FLAG "-I${CAPNP_PREFIX}/include")
    endif()
    add_custom_target(regenerate_capnp_schemas
        COMMAND ${CMAKE_COMMAND} -E env
            "LD_LIBRARY_PATH=${CAPNP_PREFIX}/lib:$ENV{LD_LIBRARY_PATH}"
            ${CAPNPC_EXECUTABLE} compile -oc++
            --src-prefix=${SCHEMA_DIR}
            ${CAPNP_INCLUDE_FLAG}
            ${SCHEMA_DIR}/math.capnp
            ${SCHEMA_DIR}/enumerations.capnp
            ${SCHEMA_DIR}/shm_synchronized_transport.capnp
            ${SCHEMA_DIR}/shm_parameter_rpc.capnp
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/generated
        COMMENT "Regenerating Cap'n Proto C++ files from schemas"
    )
endif()
