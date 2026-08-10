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

#include <cuda_runtime.h>
#include "holoscan/utils/cuda_macros.hpp"
#include "tcn_panoptic_map.hpp"

namespace tcn::ops {

namespace {

// One thread per output pixel. Loops over all M detections, keeping the one with the largest
// `priorities[j]` among those that (a) cover this pixel (`masks[j][idx] != 0`) and (b) are not
// an unknown label (`values[j] != 0`). No atomics and no ordering requirement between threads,
// since each thread owns exactly one output pixel and reads (never writes) the per-detection
// inputs.
__global__ void panoptic_paint_kernel(const uint8_t* __restrict__ masks,
                                       const uint16_t* __restrict__ values,
                                       const int32_t* __restrict__ priorities,
                                       int M, int num_pixels,
                                       uint16_t* __restrict__ out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_pixels) return;

  int32_t best_priority = -1;
  uint16_t best_value = 0;
  for (int j = 0; j < M; ++j) {
    const uint16_t v = values[j];
    if (v == 0) continue;  // unknown label: this detection never paints
    if (masks[static_cast<size_t>(j) * static_cast<size_t>(num_pixels) + idx] == 0) continue;
    const int32_t p = priorities[j];
    if (p > best_priority) {
      best_priority = p;
      best_value = v;
    }
  }
  out[idx] = best_value;  // every pixel written exactly once, 0 = background
}

}  // namespace

void launch_panoptic_paint(const uint8_t* masks,
                            const uint16_t* values,
                            const int32_t* priorities,
                            int M, int H, int W,
                            uint16_t* out,
                            cudaStream_t stream) {
  const int num_pixels = H * W;
  if (num_pixels <= 0) return;

  if (M == 0) {
    // No detections on this camera: every pixel is background. The kernel's per-pixel loop
    // over M would do nothing and leave `out` untouched, so the caller still needs a zeroed
    // buffer -- zero it here instead of requiring every caller to remember to (see design doc
    // §2 / Risks: "M = 0 must return an all-zero map without launching").
    HOLOSCAN_CUDA_CALL(cudaMemsetAsync(
        out, 0, static_cast<size_t>(num_pixels) * sizeof(uint16_t), stream));
    return;
  }

  const int threads = 256;
  const int blocks = (num_pixels + threads - 1) / threads;
  panoptic_paint_kernel<<<blocks, threads, 0, stream>>>(masks, values, priorities, M, num_pixels,
                                                          out);
}

}  // namespace tcn::ops
