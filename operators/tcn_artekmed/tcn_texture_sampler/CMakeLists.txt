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
project(tcn_texture_sampler LANGUAGES CXX CUDA)

find_package(holoscan 3.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

find_package(Eigen3 REQUIRED NO_MODULE)
# Decoder Operator
add_library(tcn_texture_sampler SHARED
        tcn_texture_sampler.cu
        ../cuda/tcn_texture_sampler_kernel.cu
)
add_library(holoscan::ops::tcn_texture_sampler ALIAS tcn_texture_sampler)

target_link_libraries(tcn_texture_sampler PUBLIC
  holoscan::core
  cuda
  PRIVATE
  Eigen3::Eigen
)
target_compile_options(tcn_texture_sampler
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
target_include_directories(tcn_texture_sampler PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

if(HOLOHUB_BUILD_PYTHON)
  add_subdirectory(python)
endif()
