# SPDX-FileCopyrightText: Copyright (c) 2026 TUM CAMP / NARVIS. All rights reserved.
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
project(tcn_labeled_pointcloud LANGUAGES CXX CUDA)

find_package(holoscan 3.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

find_package(Eigen3 REQUIRED NO_MODULE)

add_library(tcn_labeled_pointcloud SHARED
        tcn_labeled_pointcloud.cu
        ../cuda/tcn_labeled_pointcloud_kernel.cu
)
add_library(holoscan::ops::tcn_labeled_pointcloud ALIAS tcn_labeled_pointcloud)

target_link_libraries(tcn_labeled_pointcloud PUBLIC
  holoscan::core
  cuda
  PRIVATE
  Eigen3::Eigen
)
target_compile_options(tcn_labeled_pointcloud
  PRIVATE
    # --extended-lambda: the compaction predicate is a __device__ lambda passed to thrust
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)
target_include_directories(tcn_labeled_pointcloud PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

if(HOLOHUB_BUILD_PYTHON)
  add_subdirectory(python)
endif()
