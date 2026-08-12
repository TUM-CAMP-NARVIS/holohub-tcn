/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 TUM CAMP / NARVIS. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>

#include "../common/datatypes.hpp"
#include "tcn_label_sampler_kernel.cuh"   // packed-label constants, shared on purpose

/// Mark the points belonging to one class. `cls < 0` marks every non-background label, matching
/// tcn_label_sampler's "empty select_classes selects everything" convention.
void launch_select_class(const uint16_t* labels,
                         int64_t count,
                         int cls,
                         uint8_t* selected,
                         cudaStream_t stream);

/// Gather `n` points named by `indices` out of the full-resolution arrays.
void launch_gather_points(const float* positions,     ///< [count*3] xyz, world space
                          const uint16_t* labels,     ///< [count]
                          const int32_t* indices,     ///< [n] source indices, ascending
                          int64_t n,
                          float* out_positions,       ///< [n*3]
                          uint16_t* out_labels,       ///< [n]
                          cudaStream_t stream);

/// Write a single NaN point, used when a class has no points this frame. An empty tensor would
/// starve the merger downstream, and a zero position would draw a stray point at the origin; a NaN
/// vertex is culled by the rasteriser instead.
void launch_write_empty_point(float* out_positions,
                              uint16_t* out_labels,
                              cudaStream_t stream);
