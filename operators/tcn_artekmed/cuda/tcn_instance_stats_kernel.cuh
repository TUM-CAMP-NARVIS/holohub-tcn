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
  kColYaw,               ///< rotation about the vertical axis, radians, from ground-plane PCA
  kColOrientedU,         ///< extent along the yaw direction
  kColOrientedV,         ///< extent along the in-plane perpendicular
  kColOrientedUp,        ///< extent along the vertical axis (same as the AABB's)
  kColOrientedCenterX, kColOrientedCenterY, kColOrientedCenterZ,   ///< centre of the oriented box
  kInstanceStatColumns
};

/// Histogram bins per axis, used to find robust percentile bounds. 64 gives ~1.5% resolution on the
/// instance's own range -- finer than the trim thresholds need. The table is slots*3*bins*4 bytes,
/// i.e. 50 MB for the full label space, which is the price of not needing to know which labels are
/// present before binning. Sized once at start(), not per frame.
constexpr int kInstanceHistBins = 64;

/// Propagate + pointer-jump rounds in the component pre-filter. Pointer jumping shortens chains
/// geometrically, so even a structure spanning the full grid width converges in about five; the
/// margin here buys immunity to awkward shapes without a host round trip to test for convergence.
constexpr int kComponentRounds = 20;

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
  // Ground-plane second moments of the surviving points, for the yaw. Only the two horizontal axes
  // participate: the box stays axis-aligned vertically (objects in a room stand upright), so a full
  // 3D PCA would fit noise in the one direction we already know.
  float*    sumUU;       ///< [slots]      sum of u*u, u = first horizontal axis, mean-free at use
  float*    sumVV;       ///< [slots]      sum of v*v
  float*    sumUV;       ///< [slots]      sum of u*v -- the cross term that carries the orientation
  float*    yaw;         ///< [slots]      derived
  int32_t*  oMinEnc;     ///< [slots*2]    min along (yaw, perpendicular), ordered-int encoded
  int32_t*  oMaxEnc;     ///< [slots*2]    max along the same
};

/// Scratch for the connected-component pre-filter. Sized by the depth grid, not the label space, so
/// it is allocated on the first tick (when H and W are known) and reused.
struct ComponentBuffers {
  uint32_t* comp = nullptr;        ///< [H*W]   union-find parent, converged to the component root
  uint32_t* compCount = nullptr;   ///< [H*W]   points per root (indexed BY root, hence H*W wide)
  unsigned long long* best = nullptr;  ///< [slots] packed (count << 32 | root), largest per label
  uint16_t* labelsOut = nullptr;   ///< [H*W]   input labels with the losing components zeroed
  int64_t capacity = 0;            ///< allocated H*W; a larger frame triggers a realloc
};

/// Reject mask bleed by CONNECTIVITY rather than by distance: keep only the largest connected
/// component of each instance and zero the rest, writing the result to `bufs.labelsOut`.
///
/// Why connectivity. Every distance-based rule here has a breakdown point -- percentile trimming
/// removes outliers up to `trim_percentile` however far away they lie, and no further; sigma
/// clipping is worse, because the outliers inflate the sigma meant to catch them (the masking
/// effect, see this header's .cu). A real bleed population is routinely 20% of an instance, above
/// both. Connectivity has no such bound: it does not care how MANY the outliers are, only that they
/// are not attached to the object.
///
/// Why the image grid and not a voxel grid. These points come from a depth image, so they are an
/// ORGANISED cloud: 3D adjacency is 2D pixel adjacency plus a depth-continuity test. That makes this
/// a 2D component labelling over H*W instead of a voxel grid that would have to be allocated, sized
/// and hashed -- and it is the more faithful test, because mask bleed lands on a background surface
/// and is therefore separated from its object by exactly the depth discontinuity this predicate
/// looks for.
///
/// Two pixels join when they share a packed label AND their 3D separation is <= `max_gap_m`. The
/// gap is what makes this a surface test rather than a mask test: without it every pixel of one
/// mask would be one component no matter how far apart in space, which is the bug being fixed.
///
/// `min_fraction` keeps every component holding at least that share of the largest, not only the
/// single largest -- an object split by an occluder is genuinely two components, and keeping only
/// one would discard a real part. 1.0 keeps strictly the largest.
///
/// Convergence: `kComponentRounds` rounds of propagate + pointer-jump, no host synchronisation.
/// Pointer jumping shortens chains geometrically, so a 640-wide structure converges in ~5 rounds;
/// the cap is generous. A non-converged component only ever splits an object further, never merges
/// two -- so the failure mode is conservative.
void launch_component_filter(const float* positions,    ///< [H*W*3] xyz, world space
                             const uint16_t* labels,    ///< [H*W] packed panoptic, unmodified
                             int height, int width,
                             float max_gap_m,
                             float min_fraction,
                             const ComponentBuffers& bufs,
                             cudaStream_t stream);

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

/// Pass 3: count, sum, axis-aligned min/max and ground-plane second moments over only the points
/// inside the robust bounds on every axis. `up_axis` selects the vertical world axis.
void launch_instance_trimmed(const float* positions,
                             const uint16_t* labels,
                             int64_t count,
                             int up_axis,
                             const InstanceAccumulators& acc,
                             cudaStream_t stream);

/// Derive the yaw per slot from the ground-plane covariance of the surviving points.
///
/// `min_anisotropy` is the ratio the two in-plane eigenvalues must differ by before a yaw is trusted.
/// A near-circular footprint has no meaningful orientation, and fitting one to noise makes the box
/// rotate randomly frame to frame -- worse than reporting no rotation at all. Below the threshold the
/// yaw is set to 0, i.e. the oriented box degenerates to the axis-aligned one.
void launch_instance_yaw(const InstanceAccumulators& acc, int up_axis,
                         float min_anisotropy, cudaStream_t stream);

/// Pass 4: min/max of the surviving points PROJECTED onto the yaw frame, giving the oriented extents.
/// A separate pass because the projection is not knowable until the yaw is.
void launch_instance_oriented(const float* positions,
                              const uint16_t* labels,
                              int64_t count,
                              int up_axis,
                              const InstanceAccumulators& acc,
                              cudaStream_t stream);

/// Compact the non-empty slots into `rows` / `row_labels`, appending via `row_count`. Slots with
/// fewer than `min_points` trimmed points are dropped -- a handful of points is depth noise or mask
/// fringe, not an object.
void launch_instance_compact(const InstanceAccumulators& acc,
                             int up_axis,
                             int camera_index,
                             uint32_t min_points,
                             float* rows,             ///< [max_rows * kInstanceStatColumns]
                             uint16_t* row_labels,    ///< [max_rows]
                             uint32_t* row_count,     ///< [1]
                             uint32_t max_rows,
                             cudaStream_t stream);
