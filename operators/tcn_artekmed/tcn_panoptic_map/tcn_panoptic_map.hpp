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

// Erode every instance region of a packed panoptic map by `radius` pixels.
//
// Motivation: the map is sampled through the depth image's texcoords (`tcn_label_sampler`), so a
// mask that overshoots its object by a few COLOUR pixels labels background depth pixels as that
// object. Those points are real, finite and metres away, and they are what `tcn_instance_stats`'
// percentile trim spends its breakdown point on. Eroding before the lookup removes the bleed at
// its source rather than paying for it downstream.
//
// A pixel keeps its packed label only if every pixel in the (2*radius+1)^2 window carries the
// SAME label; otherwise it becomes 0 (background). Because labels partition the image, that one
// test erodes every instance at once -- no per-label pass -- and it also opens a seam between two
// instances that touch.
//
//   in:       (H, W) uint16 device buffer, the packed (class << 8) | instance map.
//   scratch:  (H, W) uint16 device buffer, caller-owned. Holds the intermediate of the separable
//             pass; its contents on entry are irrelevant and on exit meaningless. MUST NOT alias
//             `in` or `out`.
//   out:      (H, W) uint16 device buffer, written IN FULL (including at radius <= 0), so it does
//             not need pre-zeroing. MAY alias `in` -- after the horizontal pass `in` is never
//             read again.
//   radius:   erosion radius in pixels, in the map's own (colour) resolution. <= 0 is the
//             identity and launches no kernel. Note the depth grid the labels are sampled onto is
//             typically ~3x coarser, so a radius below ~3 barely moves a depth pixel.
//   stream:   CUDA stream both passes are enqueued on.
//
// Separable: the square window factors into a horizontal then a vertical pass. After the first
// pass a pixel is already either its own label or 0, so the vertical all-equal test composes to
// exactly the square window. `erode_panoptic_np` (tcn_langsam/helpers.py) is the reference
// implementation and tests/test_panoptic_erosion.py pins the equivalence against a brute-force
// square oracle.
//
// Borders are CLAMPED, not treated as background: an object running off the edge of the frame
// keeps its pixels there instead of being shaved from that side.
void launch_panoptic_erode(const uint16_t* in,
                            uint16_t* scratch,
                            uint16_t* out,
                            int H, int W, int radius,
                            cudaStream_t stream);

}  // namespace tcn::ops
