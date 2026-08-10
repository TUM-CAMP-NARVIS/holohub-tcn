/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <pybind11/pybind11.h>

#include <cstdint>
#include <cuda_runtime.h>

#include "../tcn_panoptic_map.hpp"

using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace tcn::ops {

namespace {

// Pointers cross the pybind boundary as plain integers (cupy's `.data.ptr` / a CUDA stream
// handle), not as any tensor/DLPack type -- see design doc §3, "Packaging": this keeps the
// binding free of any holoscan::Tensor / DLPack dependency, so it works from any array module
// (cupy today) that exposes a raw device pointer.
void build_panoptic_map_cuda(uintptr_t masks_ptr,
                              uintptr_t values_ptr,
                              uintptr_t priorities_ptr,
                              int M, int H, int W,
                              uintptr_t out_ptr,
                              uintptr_t stream_ptr) {
  launch_panoptic_paint(reinterpret_cast<const uint8_t*>(masks_ptr),
                         reinterpret_cast<const uint16_t*>(values_ptr),
                         reinterpret_cast<const int32_t*>(priorities_ptr),
                         M, H, W,
                         reinterpret_cast<uint16_t*>(out_ptr),
                         reinterpret_cast<cudaStream_t>(stream_ptr));
}

}  // namespace

PYBIND11_MODULE(_tcn_panoptic_map, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK TCN Panoptic Map Python Bindings
        ----------------------------------------------
        .. currentmodule:: _tcn_panoptic_map
        .. autosummary::
           :toctree: _generate
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  // A free function, not an Operator: this kernel runs inline inside LangSamBatchOp in the
  // monolithic (default) path, so a callable is usable there directly. See design doc §3.
  // Precedent for a free function in this tree: `discover_shm` in
  // tcn_shm_subscriber/python/shm_subscriber_op.cpp.
  m.def("build_panoptic_map_cuda",
        &build_panoptic_map_cuda,
        "masks_ptr"_a,
        "values_ptr"_a,
        "priorities_ptr"_a,
        "M"_a,
        "H"_a,
        "W"_a,
        "out_ptr"_a,
        "stream_ptr"_a,
        R"pbdoc(
        Fused panoptic-map paint kernel launcher.

        Per output pixel, the covering detection with the LARGEST `priorities` value wins;
        detections with `values == 0` (unknown label) never paint. Equivalent to painting each
        detection's mask in ascending-score order and letting the last write win, including for
        tied scores (`priorities` is a unique permutation of 0..M-1, computed on the host by
        `plan_panoptic_paint`).

        All pointer arguments are raw CUDA device addresses (e.g. cupy's `.data.ptr`), passed as
        plain integers -- this binding has no DLPack / holoscan::Tensor dependency.

        Parameters
        ----------
        masks_ptr : int
            Device pointer to a contiguous (M, H, W) row-major uint8 buffer.
        values_ptr : int
            Device pointer to a (M,) uint16 buffer: packed (class_id << 8) | instance_id, 0 =
            skip this detection.
        priorities_ptr : int
            Device pointer to a (M,) int32 buffer: unique per detection, larger wins.
        M : int
            Number of detections. M == 0 zero-fills `out_ptr` without launching a kernel.
        H : int
            Output map height.
        W : int
            Output map width.
        out_ptr : int
            Device pointer to a (H, W) uint16 buffer. Written IN FULL -- no pre-zeroing needed.
        stream_ptr : int
            CUDA stream handle (e.g. `cupy.cuda.get_current_stream().ptr`) the work is enqueued
            on.
        )pbdoc");
}  // PYBIND11_MODULE

}  // namespace tcn::ops
