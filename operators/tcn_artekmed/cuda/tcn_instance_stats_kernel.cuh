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

/// Histogram bins per axis, used to find robust percentile bounds. 64 gives ~1.5% resolution on the
/// instance's own range -- finer than the trim thresholds need. The table is slots*3*bins*4 bytes,
/// i.e. 50 MB for the full label space, which is the price of not needing to know which labels are
/// present before binning. Sized once at start(), not per frame.
constexpr int kInstanceHistBins = 64;

/// Device-side accumulators, all sized [kInstanceSlots] (or x3 where noted). Allocated once and
/// reused.
struct InstanceAccumulators {
  // Raw statistics over every point of the instance.
  uint32_t* count1;      ///< [slots]      raw point count
  float*    sum1;        ///< [slots*3]    raw sum of positions
  float*    sqsum1;      ///< [slots*3]    raw sum of squares, for the reported (untrimmed) sigma
  float*    sigmaRaw;    ///< [slots*3]    untrimmed spread -- the mask-bleed indicator
  int32_t*  rawMinEnc;   ///< [slots*3]    raw min, the histogram's lower edge
  int32_t*  rawMaxEnc;   ///< [slots*3]    raw max, the histogram's upper edge
  // Robust bounds, from percentiles of a per-axis histogram over [rawMin, rawMax].
  uint32_t* hist;        ///< [slots*3*bins]
  float*    loBound;     ///< [slots*3]    lower percentile bound used for trimming
  float*    hiBound;     ///< [slots*3]    upper percentile bound
  // Aggregates over the points that survive the trim.
  uint32_t* count2;      ///< [slots]
  float*    sum2;        ///< [slots*3]
  int32_t*  minEnc;      ///< [slots*3]    trimmed min
  int32_t*  maxEnc;      ///< [slots*3]    trimmed max
};

/// Zero the accumulators (min/max are set to the encoding's extremes, not to 0).
void launch_instance_reset(const InstanceAccumulators& acc, cudaStream_t stream);

/// Pass 1: raw count, sum, sum-of-squares and min/max per label. Label 0 is skipped.
void launch_instance_pass1(const float* positions,      ///< [count*3] xyz, world space
                           const uint16_t* labels,      ///< [count] packed panoptic
                           int64_t count,
                           const InstanceAccumulators& acc,
                           cudaStream_t stream);

/// Record the untrimmed sigma (reported as the bleed indicator, never used for trimming).
void launch_instance_finalize_raw(const InstanceAccumulators& acc, cudaStream_t stream);

/// Pass 2: per-axis histogram of every point over its instance's raw range.
void launch_instance_histogram(const float* positions,
                               const uint16_t* labels,
                               int64_t count,
                               const InstanceAccumulators& acc,
                               cudaStream_t stream);

/// Turn each histogram into a lower/upper percentile bound, widened by `margin` of the kept range.
///
/// Percentiles rather than standard deviations, because sigma cannot survive a large outlier
/// population: a blob holding a quarter of the points inflates sigma enough that the +-k*sigma window
/// contains the blob itself -- the masking effect -- and iterating cannot escape that, since the first
/// pass already rejects nothing. A percentile bound is defined by point COUNT, so it is unmoved by how
/// far away the outliers are, and tolerates up to `trim_percentile` of them on each side by
/// construction.
void launch_instance_bounds(const InstanceAccumulators& acc,
                            float trim_percentile,   ///< e.g. 0.02 -> keep the 2nd..98th percentile
                            float margin,            ///< widen by this fraction of the kept range
                            float min_range,         ///< floor, so a flat axis keeps its own points
                            cudaStream_t stream);

/// Pass 3: count, sum, min and max over only the points inside the robust bounds on every axis.
void launch_instance_trimmed(const float* positions,
                             const uint16_t* labels,
                             int64_t count,
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
