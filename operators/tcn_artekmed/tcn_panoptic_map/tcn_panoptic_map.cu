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

// One thread per pixel, one axis. A pixel keeps its packed label only if every pixel in the
// `radius`-neighbourhood along this axis carries the SAME label; otherwise it becomes 0.
//
// Because labels partition the image, that single test erodes every instance region at once --
// there is no per-label pass. It also opens a seam between two instances that touch, which is the
// same bleed problem seen from the other side.
//
// Borders are CLAMPED (the window truncates at the edge) rather than treated as background, so an
// object running off the side of the frame is not eaten away from that side.
//
// Both passes write every output pixel exactly once, so neither needs `out` pre-zeroed.
template <bool kHorizontal>
__global__ void panoptic_erode_kernel(const uint16_t* __restrict__ in,
                                      uint16_t* __restrict__ out,
                                      int H, int W, int radius) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= H * W) return;

  const int x = idx % W;
  const int y = idx / W;
  const uint16_t v = in[idx];

  const int centre = kHorizontal ? x : y;
  const int extent = kHorizontal ? W : H;
  const int lo = max(0, centre - radius);
  const int hi = min(extent - 1, centre + radius);
  const int stride = kHorizontal ? 1 : W;
  const int base = kHorizontal ? (y * W) : x;

  for (int i = lo; i <= hi; ++i) {
    if (in[base + i * stride] != v) {
      out[idx] = 0;
      return;
    }
  }
  out[idx] = v;
}

}  // namespace

void launch_panoptic_erode(const uint16_t* in,
                            uint16_t* scratch,
                            uint16_t* out,
                            int H, int W, int radius,
                            cudaStream_t stream) {
  const int num_pixels = H * W;
  if (num_pixels <= 0) return;
  const size_t bytes = static_cast<size_t>(num_pixels) * sizeof(uint16_t);

  if (radius <= 0) {
    // Identity, but `out` is still written IN FULL -- the caller's contract does not change with
    // the radius, so a caller that passes a fresh uninitialised buffer stays correct at radius 0.
    // Skipped entirely (not even this copy) when `in == out`.
    if (in != out) { HOLOSCAN_CUDA_CALL(cudaMemcpyAsync(out, in, bytes,
                                                        cudaMemcpyDeviceToDevice, stream)); }
    return;
  }

  const int threads = 256;
  const int blocks = (num_pixels + threads - 1) / threads;
  // Separable: the square (2r+1)^2 window factors into a horizontal then a vertical pass. After
  // the first pass a pixel is already either its own label or 0, so requiring the vertical run to
  // be all-equal composes to exactly the square window -- `erode_panoptic_np` in
  // tcn_langsam/helpers.py is the reference and tests/test_panoptic_erosion.py pins the
  // equivalence against a brute-force square oracle.
  //
  // `out` may alias `in`: after the horizontal pass `in` is never read again.
  panoptic_erode_kernel<true><<<blocks, threads, 0, stream>>>(in, scratch, H, W, radius);
  panoptic_erode_kernel<false><<<blocks, threads, 0, stream>>>(scratch, out, H, W, radius);
}

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
