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

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace tcn::ops {

// Fused panoptic-map paint. Replaces the per-detection cupy boolean-mask assignment
// (`pmap[masks[j] > 0] = v`, once per detection) whose `nonzero()` forces a device
// synchronisation per detection -- see docs/specs/2026-08-10-panoptic-cuda-design.md.
//
// Reformulation (design doc §1): painting `masks[j]` in ascending-score order so the most
// confident detection wins an overlap is equivalent to, per pixel, selecting the covering
// detection with the highest score. `priorities[j]` is detection j's index in that
// ascending-score `argsort` order (computed on the host, unique per detection), so "the
// covering detection with the LARGEST priority wins" reproduces the legacy "last painted wins"
// behaviour exactly, including for tied scores. One thread per pixel, loop over M, no atomics,
// no ordering required.
//
//   masks:       (M, H, W) row-major uint8 device buffer, contiguous. masks[j*H*W + idx] != 0
//                means detection j covers pixel `idx`.
//   values:      (M,) uint16 device buffer. Packed (class_id << 8) | instance_id; 0 means
//                "unknown label" and this detection never paints (skipped even if it covers
//                the pixel and has the highest priority).
//   priorities:  (M,) int32 device buffer. Unique per detection (it is a permutation of
//                0..M-1); LARGER wins an overlap.
//   M, H, W:     detection count and output map shape.
//   out:         (H, W) uint16 device buffer. Written IN FULL by this call -- every pixel is
//                written exactly once (either a winning `values[j]` or 0), so the caller does
//                NOT need to pre-zero `out` before calling.
//   stream:      CUDA stream the paint (and, for M == 0, the zero-fill) is enqueued on.
//
// M == 0 (no detections on this camera) is handled without launching the kernel: `out` is
// zeroed via `cudaMemsetAsync` on `stream` so the caller still gets a valid all-zero map. This
// mirrors `build_panoptic_map(None, [], None, ...)`'s early return in the cupy/numpy path.
void launch_panoptic_paint(const uint8_t* masks,
                            const uint16_t* values,
                            const int32_t* priorities,
                            int M, int H, int W,
                            uint16_t* out,
                            cudaStream_t stream);

}  // namespace tcn::ops
