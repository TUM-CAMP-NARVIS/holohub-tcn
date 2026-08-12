/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 TUM CAMP / NARVIS. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>

#include "../common/datatypes.hpp"
#include "tcn_label_sampler_kernel.cuh"   // packed-label constants, shared on purpose

/// Slots in the accumulator table: one per possible packed uint16 label. Sizing the table by the
/// label space rather than by the number of instances present is what lets the whole reduction run
/// without first discovering how many instances there are -- no pre-pass, no host round trip.
constexpr int kInstanceSlots = 65536;

/// Columns of one output row. Kept in one place because the Python consumer indexes them positionally
/// (see tcn_object_tracking); adding a column means updating both.
enum InstanceStatColumn {
  kColCameraIndex = 0,   ///< which camera this observation came from
  kColCount,             ///< points surviving the sigma trim
  kColCentroidX, kColCentroidY, kColCentroidZ,
  kColMinX, kColMinY, kColMinZ,
  kColMaxX, kColMaxY, kColMaxZ,
  kColSigmaX, kColSigmaY, kColSigmaZ,
  kInstanceStatColumns
};

/// Device-side accumulators, all sized [kInstanceSlots] (or x3 where noted). Allocated once and
/// reused; pass 1 fills the raw moments, pass 2 the trimmed aggregates.
struct InstanceAccumulators {
  uint32_t* count1;      ///< [slots]      raw point count
  float*    sum1;        ///< [slots*3]    raw sum of positions
  float*    sqsum1;      ///< [slots*3]    raw sum of squares, for sigma
  float*    mean;        ///< [slots*3]    derived from pass 1
  float*    sigma;       ///< [slots*3]    derived from pass 1
  uint32_t* count2;      ///< [slots]      trimmed point count
  float*    sum2;        ///< [slots*3]    trimmed sum of positions
  int32_t*  minEnc;      ///< [slots*3]    trimmed min, ordered-int encoded for atomicMin
  int32_t*  maxEnc;      ///< [slots*3]    trimmed max, ordered-int encoded for atomicMax
};

/// Zero the accumulators (min/max are set to the encoding's extremes, not to 0).
void launch_instance_reset(const InstanceAccumulators& acc, cudaStream_t stream);

/// Pass 1: raw count, sum and sum-of-squares per label. Label 0 (background) is skipped.
void launch_instance_pass1(const float* positions,      ///< [count*3] xyz, world space
                           const uint16_t* labels,      ///< [count] packed panoptic
                           int64_t count,
                           const InstanceAccumulators& acc,
                           cudaStream_t stream);

/// Derive mean and sigma per slot from pass 1.
void launch_instance_finalize1(const InstanceAccumulators& acc, cudaStream_t stream);

/// Pass 2: count, sum, min and max over only the points within `sigma_k` standard deviations of the
/// mean on every axis. `sigma_floor` keeps a perfectly flat instance (sigma ~ 0 on some axis, e.g. a
/// wall patch) from rejecting all of its own points.
void launch_instance_pass2(const float* positions,
                           const uint16_t* labels,
                           int64_t count,
                           float sigma_k,
                           float sigma_floor,
                           const InstanceAccumulators& acc,
                           cudaStream_t stream);

/// Compact the non-empty slots into `rows` / `row_labels`, appending via `row_count`. Slots with
/// fewer than `min_points` trimmed points are dropped -- a handful of points is depth noise or mask
/// fringe, not an object.
void launch_instance_compact(const InstanceAccumulators& acc,
                             int camera_index,
                             uint32_t min_points,
                             float* rows,             ///< [max_rows * kInstanceStatColumns]
                             uint16_t* row_labels,    ///< [max_rows]
                             uint32_t* row_count,     ///< [1]
                             uint32_t max_rows,
                             cudaStream_t stream);
