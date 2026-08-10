# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

include(pybind11_add_holohub_module)

# CLASS_NAME feeds the generated __init__.py's `from ._tcn_panoptic_map import <CLASS_NAME>`.
# This module exports a free function, not a class -- set it to the exported function name
# (see python/tcn_panoptic_map.cpp: m.def("build_panoptic_map_cuda", ...)).
pybind11_add_holohub_module(
    CPP_CMAKE_TARGET tcn_panoptic_map
    CLASS_NAME "build_panoptic_map_cuda"
    SOURCES tcn_panoptic_map.cpp
)
