/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 TUM CAMP / NARVIS. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Per-instance reduction over a labeled point grid: count, centroid, robust axis-aligned box and
 * per-axis spread for every packed panoptic label present.
 *
 * This is a reduction BY KEY, not a clustering problem: the masks have already segmented the points,
 * so every point carries the instance it belongs to. What remains is outlier rejection, because a mask
 * edge lands on whatever surface is behind the object and those points are real, finite and wrong.
 *
 * Outliers are rejected by PERCENTILE, not by standard deviation. Sigma cannot survive a large outlier
 * population: a blob holding a quarter of an instance's points inflates sigma enough that the
 * +-k*sigma window contains the blob itself (the masking effect), and iterating the clip cannot escape
 * that -- the first pass rejects nothing, so it is already at a fixed point. Worse, iterating a tight
 * unimodal cluster erodes it, since each pass discards the tails of what the previous pass kept.
 * A percentile bound is defined by point COUNT, so it is indifferent to how far away the outliers lie.
 */
#include <cuda_runtime.h>
#include <cmath>

#include "tcn_instance_stats_kernel.cuh"

namespace {

constexpr int kBlock = 256;

int64_t grid_for(int64_t n) { return (n + kBlock - 1) / kBlock; }

/// Monotonic float -> int mapping, so integer atomicMin/atomicMax order floats correctly.
/// Non-negative floats already compare correctly as signed ints; negatives compare in reverse, which
/// flipping the low 31 bits fixes.
__device__ __forceinline__ int32_t float_to_ordered(float f) {
  int32_t i = __float_as_int(f);
  return (i >= 0) ? i : (i ^ 0x7FFFFFFF);
}

__device__ __forceinline__ float ordered_to_float(int32_t i) {
  return (i >= 0) ? __int_as_float(i) : __int_as_float(i ^ 0x7FFFFFFF);
}

// ── connected-component pre-filter ───────────────────────────────────────────────────────────────
//
// See the .cuh for why this is connectivity rather than distance, and why it runs on the image grid
// rather than a voxel grid. Everything below writes only into ComponentBuffers; the input labels are
// never modified.

constexpr uint32_t kNoComponent = 0xFFFFFFFFu;

__global__ void component_init_kernel(const float* __restrict__ positions,
                                      const uint16_t* __restrict__ labels,
                                      int64_t count,
                                      uint32_t* __restrict__ comp,
                                      uint32_t* __restrict__ compCount) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  compCount[i] = 0u;
  const uint16_t label = labels[i];
  // Background and non-finite pixels take no part: they must not bridge two real components.
  if (label == 0 || !isfinite(positions[3 * i + 0]) || !isfinite(positions[3 * i + 1]) ||
      !isfinite(positions[3 * i + 2])) {
    comp[i] = kNoComponent;
    return;
  }
  comp[i] = static_cast<uint32_t>(i);          // every pixel starts as its own root
}

/// One propagation round: adopt the smallest root among the four-neighbours that share this pixel's
/// label AND lie within `max_gap_m` of it in 3D.
///
/// Four-connectivity, not eight: a diagonal-only link is a single-pixel bridge, and mask fringe is
/// exactly where such bridges occur -- letting one join the bleed to the object would defeat the
/// whole filter.
__global__ void component_propagate_kernel(const float* __restrict__ positions,
                                           const uint16_t* __restrict__ labels,
                                           int height, int width,
                                           float max_gap_sq,
                                           uint32_t* __restrict__ comp) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= height * width) return;
  const uint32_t mine = comp[idx];
  if (mine == kNoComponent) return;

  const int x = idx % width;
  const int y = idx / width;
  const uint16_t label = labels[idx];
  const float p[3] = {positions[3 * idx + 0], positions[3 * idx + 1], positions[3 * idx + 2]};

  uint32_t best = mine;
  const int dx[4] = {-1, 1, 0, 0};
  const int dy[4] = {0, 0, -1, 1};
  for (int k = 0; k < 4; ++k) {
    const int nx = x + dx[k], ny = y + dy[k];
    if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
    const int n = ny * width + nx;
    if (labels[n] != label) continue;
    const uint32_t other = comp[n];
    if (other == kNoComponent) continue;
    const float d0 = positions[3 * n + 0] - p[0];
    const float d1 = positions[3 * n + 1] - p[1];
    const float d2 = positions[3 * n + 2] - p[2];
    if (d0 * d0 + d1 * d1 + d2 * d2 > max_gap_sq) continue;   // a depth step: different surface
    if (other < best) best = other;
  }
  if (best != mine) atomicMin(&comp[idx], best);
}

/// Pointer jumping: replace each pixel's root by its root's root. Halves chain length per
/// application, which is what turns O(diameter) propagation into O(log diameter).
__global__ void component_compress_kernel(uint32_t* __restrict__ comp, int64_t count) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  uint32_t r = comp[i];
  if (r == kNoComponent) return;
  uint32_t rr = comp[r];
  if (rr == kNoComponent) return;              // cannot happen for a valid root, but never chase it
  comp[i] = rr;
}

__global__ void component_count_kernel(const uint32_t* __restrict__ comp,
                                       int64_t count,
                                       uint32_t* __restrict__ compCount) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const uint32_t r = comp[i];
  if (r == kNoComponent) return;
  atomicAdd(&compCount[r], 1u);
}

/// Largest component per label, as a packed (count << 32 | root) so one 64-bit atomicMax settles
/// both the winning size and its identity. Ties break on the larger root index, which is arbitrary
/// but deterministic -- what matters is that every pixel agrees on the same winner.
__global__ void component_best_kernel(const uint16_t* __restrict__ labels,
                                      const uint32_t* __restrict__ comp,
                                      const uint32_t* __restrict__ compCount,
                                      int64_t count,
                                      unsigned long long* __restrict__ best) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const uint32_t r = comp[i];
  if (r == kNoComponent) return;
  const uint16_t label = labels[i];
  if (label == 0) return;
  const unsigned long long key =
      (static_cast<unsigned long long>(compCount[r]) << 32) | static_cast<unsigned long long>(r);
  atomicMax(&best[label], key);
}

__global__ void component_select_kernel(const uint16_t* __restrict__ labels,
                                        const uint32_t* __restrict__ comp,
                                        const uint32_t* __restrict__ compCount,
                                        const unsigned long long* __restrict__ best,
                                        int64_t count,
                                        float min_fraction,
                                        uint16_t* __restrict__ labelsOut) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const uint16_t label = labels[i];
  const uint32_t r = comp[i];
  if (label == 0 || r == kNoComponent) { labelsOut[i] = 0u; return; }
  const uint32_t largest = static_cast<uint32_t>(best[label] >> 32);
  if (largest == 0u) { labelsOut[i] = 0u; return; }
  // Keep every component holding at least `min_fraction` of the largest, not only the largest
  // itself: an object split by an occluder is genuinely two components and dropping one would
  // discard a real part of it. At 1.0 the comparison keeps exactly the winner.
  const float share = static_cast<float>(compCount[r]) / static_cast<float>(largest);
  labelsOut[i] = (share >= min_fraction) ? label : static_cast<uint16_t>(0);
}

/// The half of the reset that a `cudaMemsetAsync` cannot do: the min/max sentinels are
/// 0x7FFFFFFF / 0x80000000 under the ordered-int encoding, and neither is a repeated byte.
///
/// Indexed FLAT over the arrays rather than per slot, so consecutive threads write consecutive
/// words. That is the whole point -- see `launch_instance_reset`.
__global__ void reset_sentinels_kernel(InstanceAccumulators acc) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kInstanceSlots * 3) return;
  acc.rawMinEnc[i] = 0x7FFFFFFF;                        // +inf under the ordered encoding
  acc.rawMaxEnc[i] = static_cast<int32_t>(0x80000000);   // -inf
  acc.minEnc[i] = 0x7FFFFFFF;
  acc.maxEnc[i] = static_cast<int32_t>(0x80000000);
  // The oriented arrays are [slots*2], so the first two thirds of the same launch cover them.
  if (i < kInstanceSlots * 2) {
    acc.oMinEnc[i] = 0x7FFFFFFF;
    acc.oMaxEnc[i] = static_cast<int32_t>(0x80000000);
  }
}

__global__ void pass1_kernel(const float* __restrict__ positions,
                             const uint16_t* __restrict__ labels,
                             int64_t count,
                             InstanceAccumulators acc) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const uint16_t label = labels[i];
  if (label == 0) return;                       // background, and unlabeled depth pixels
  const float p[3] = {positions[3 * i + 0], positions[3 * i + 1], positions[3 * i + 2]};
  // Defensive: a labeled pixel should always have valid depth (an invalid sample yields a NaN
  // texcoord, hence no label), but a non-finite position would poison every aggregate it reaches.
  if (!isfinite(p[0]) || !isfinite(p[1]) || !isfinite(p[2])) return;

  atomicAdd(&acc.count1[label], 1u);
  for (int d = 0; d < 3; ++d) {
    atomicAdd(&acc.sum1[3 * label + d], p[d]);
    atomicAdd(&acc.sqsum1[3 * label + d], p[d] * p[d]);
    atomicMin(&acc.rawMinEnc[3 * label + d], float_to_ordered(p[d]));
    atomicMax(&acc.rawMaxEnc[3 * label + d], float_to_ordered(p[d]));
  }
}

__global__ void finalize_raw_kernel(InstanceAccumulators acc) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= kInstanceSlots) return;
  const uint32_t n = acc.count1[s];
  if (n == 0u) return;
  const float inv = 1.f / static_cast<float>(n);
  for (int d = 0; d < 3; ++d) {
    const int k = 3 * s + d;
    const float mean = acc.sum1[k] * inv;
    // var = E[x^2] - E[x]^2, clamped: catastrophic cancellation on a tight cluster far from the
    // origin can make this a small negative number, and sqrt of that is NaN.
    const float var = fmaxf(acc.sqsum1[k] * inv - mean * mean, 0.f);
    acc.sigmaRaw[k] = sqrtf(var);
  }
}

__global__ void histogram_kernel(const float* __restrict__ positions,
                                 const uint16_t* __restrict__ labels,
                                 int64_t count,
                                 InstanceAccumulators acc) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const uint16_t label = labels[i];
  if (label == 0) return;
  const float p[3] = {positions[3 * i + 0], positions[3 * i + 1], positions[3 * i + 2]};
  if (!isfinite(p[0]) || !isfinite(p[1]) || !isfinite(p[2])) return;

  for (int d = 0; d < 3; ++d) {
    const int k = 3 * label + d;
    const float lo = ordered_to_float(acc.rawMinEnc[k]);
    const float hi = ordered_to_float(acc.rawMaxEnc[k]);
    const float span = hi - lo;
    int bin = 0;
    if (span > 0.f) {
      bin = static_cast<int>((p[d] - lo) / span * kInstanceHistBins);
      bin = max(0, min(bin, kInstanceHistBins - 1));
    }
    atomicAdd(&acc.hist[k * kInstanceHistBins + bin], 1u);
  }
}

__global__ void bounds_kernel(InstanceAccumulators acc, float trim_percentile, float margin,
                              float min_range) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= kInstanceSlots) return;
  const uint32_t n = acc.count1[s];
  if (n == 0u) return;

  const uint32_t target = static_cast<uint32_t>(trim_percentile * static_cast<float>(n));
  for (int d = 0; d < 3; ++d) {
    const int k = 3 * s + d;
    const float lo = ordered_to_float(acc.rawMinEnc[k]);
    const float hi = ordered_to_float(acc.rawMaxEnc[k]);
    const float span = hi - lo;
    if (span <= 0.f) {                     // every point identical on this axis
      acc.loBound[k] = lo - min_range;
      acc.hiBound[k] = hi + min_range;
      continue;
    }
    const float bin_width = span / kInstanceHistBins;
    const uint32_t* h = &acc.hist[k * kInstanceHistBins];

    // Walk in from each end until `target` points have been passed. The bound is the far edge of the
    // bin where that happens, so a bin is never partially excluded.
    uint32_t acc_lo = 0;
    int b_lo = 0;
    while (b_lo < kInstanceHistBins - 1 && acc_lo + h[b_lo] <= target) {
      acc_lo += h[b_lo];
      ++b_lo;
    }
    uint32_t acc_hi = 0;
    int b_hi = kInstanceHistBins - 1;
    while (b_hi > 0 && acc_hi + h[b_hi] <= target) {
      acc_hi += h[b_hi];
      --b_hi;
    }
    if (b_hi < b_lo) { b_hi = b_lo; }      // degenerate: everything in one bin

    float bound_lo = lo + static_cast<float>(b_lo) * bin_width;
    float bound_hi = lo + static_cast<float>(b_hi + 1) * bin_width;

    // Widen slightly: the bound sits on a bin edge, and a margin keeps the object's genuine extremes
    // (which are what the box should report) from being shaved off by binning resolution alone.
    const float kept = fmaxf(bound_hi - bound_lo, min_range);
    bound_lo -= margin * kept;
    bound_hi += margin * kept;
    acc.loBound[k] = bound_lo;
    acc.hiBound[k] = bound_hi;
  }
}

/// The two horizontal axes, given the vertical one.
__device__ __forceinline__ void plane_axes(int up_axis, int* u, int* v) {
  *u = (up_axis == 0) ? 1 : 0;
  *v = (up_axis == 2) ? 1 : 2;
}

__global__ void trimmed_kernel(const float* __restrict__ positions,
                               const uint16_t* __restrict__ labels,
                               int64_t count,
                               int up_axis,
                               InstanceAccumulators acc) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const uint16_t label = labels[i];
  if (label == 0) return;
  const float p[3] = {positions[3 * i + 0], positions[3 * i + 1], positions[3 * i + 2]};
  if (!isfinite(p[0]) || !isfinite(p[1]) || !isfinite(p[2])) return;

  // Reject per axis rather than by distance: an axis-aligned box is what this produces, so trimming
  // per axis is what actually tightens it.
  for (int d = 0; d < 3; ++d) {
    const int k = 3 * label + d;
    if (p[d] < acc.loBound[k] || p[d] > acc.hiBound[k]) return;
  }

  atomicAdd(&acc.count2[label], 1u);
  for (int d = 0; d < 3; ++d) {
    atomicAdd(&acc.sum2[3 * label + d], p[d]);
    atomicMin(&acc.minEnc[3 * label + d], float_to_ordered(p[d]));
    atomicMax(&acc.maxEnc[3 * label + d], float_to_ordered(p[d]));
  }
  // Raw second moments in the ground plane; the mean is subtracted later, from sum2.
  int ua, va;
  plane_axes(up_axis, &ua, &va);
  atomicAdd(&acc.sumUU[label], p[ua] * p[ua]);
  atomicAdd(&acc.sumVV[label], p[va] * p[va]);
  atomicAdd(&acc.sumUV[label], p[ua] * p[va]);
}

__global__ void yaw_kernel(InstanceAccumulators acc, int up_axis, float min_anisotropy) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= kInstanceSlots) return;
  const uint32_t n = acc.count2[s];
  acc.yaw[s] = 0.f;
  if (n < 3u) return;                       // fewer than three points has no orientation

  int ua, va;
  plane_axes(up_axis, &ua, &va);
  const float inv = 1.f / static_cast<float>(n);
  const float mu = acc.sum2[3 * s + ua] * inv;
  const float mv = acc.sum2[3 * s + va] * inv;
  const float cuu = fmaxf(acc.sumUU[s] * inv - mu * mu, 0.f);
  const float cvv = fmaxf(acc.sumVV[s] * inv - mv * mv, 0.f);
  const float cuv = acc.sumUV[s] * inv - mu * mv;

  // Eigenvalues of the symmetric 2x2 [[cuu, cuv], [cuv, cvv]].
  const float tr = cuu + cvv;
  const float diff = sqrtf(fmaxf((cuu - cvv) * (cuu - cvv) + 4.f * cuv * cuv, 0.f));
  const float l1 = 0.5f * (tr + diff);      // major
  const float l2 = 0.5f * (tr - diff);      // minor
  // A near-circular footprint has no meaningful orientation. Fitting one to noise makes the box spin
  // frame to frame, which is worse than reporting no rotation, so below the threshold yaw stays 0 and
  // the oriented box degenerates to the axis-aligned one.
  if (l1 <= 0.f || l2 < 0.f || l1 < min_anisotropy * fmaxf(l2, 1e-9f)) return;

  acc.yaw[s] = 0.5f * atan2f(2.f * cuv, cuu - cvv);
}

__global__ void oriented_kernel(const float* __restrict__ positions,
                                const uint16_t* __restrict__ labels,
                                int64_t count,
                                int up_axis,
                                InstanceAccumulators acc) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const uint16_t label = labels[i];
  if (label == 0) return;
  const float p[3] = {positions[3 * i + 0], positions[3 * i + 1], positions[3 * i + 2]};
  if (!isfinite(p[0]) || !isfinite(p[1]) || !isfinite(p[2])) return;
  // Same membership test as the trimmed pass: the oriented extents must describe the SAME point set
  // the axis-aligned box and the yaw describe, or the two boxes would not be of one object.
  for (int d = 0; d < 3; ++d) {
    const int k = 3 * label + d;
    if (p[d] < acc.loBound[k] || p[d] > acc.hiBound[k]) return;
  }

  int ua, va;
  plane_axes(up_axis, &ua, &va);
  const float c = cosf(acc.yaw[label]), sn = sinf(acc.yaw[label]);
  const float proj_u =  c * p[ua] + sn * p[va];
  const float proj_v = -sn * p[ua] + c * p[va];
  atomicMin(&acc.oMinEnc[2 * label + 0], float_to_ordered(proj_u));
  atomicMax(&acc.oMaxEnc[2 * label + 0], float_to_ordered(proj_u));
  atomicMin(&acc.oMinEnc[2 * label + 1], float_to_ordered(proj_v));
  atomicMax(&acc.oMaxEnc[2 * label + 1], float_to_ordered(proj_v));
}

__global__ void compact_kernel(InstanceAccumulators acc,
                               int up_axis,
                               int camera_index,
                               uint32_t min_points,
                               float* __restrict__ rows,
                               uint16_t* __restrict__ row_labels,
                               uint32_t* __restrict__ row_count,
                               uint32_t max_rows) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= kInstanceSlots) return;
  const uint32_t n = acc.count2[s];
  if (n < min_points) return;

  const uint32_t slot = atomicAdd(row_count, 1u);
  if (slot >= max_rows) return;                 // overflow is reported by the host from row_count

  float* r = rows + static_cast<size_t>(slot) * kInstanceStatColumns;
  const float inv = 1.f / static_cast<float>(n);
  r[kColCameraIndex] = static_cast<float>(camera_index);
  r[kColCount] = static_cast<float>(n);
  for (int d = 0; d < 3; ++d) {
    r[kColCentroidX + d] = acc.sum2[3 * s + d] * inv;
    r[kColMinX + d] = ordered_to_float(acc.minEnc[3 * s + d]);
    r[kColMaxX + d] = ordered_to_float(acc.maxEnc[3 * s + d]);
    r[kColSigmaX + d] = acc.sigmaRaw[3 * s + d];   // spread BEFORE trimming: the bleed indicator
  }

  int ua, va;
  plane_axes(up_axis, &ua, &va);
  const float yaw = acc.yaw[s];
  const float lo_u = ordered_to_float(acc.oMinEnc[2 * s + 0]);
  const float hi_u = ordered_to_float(acc.oMaxEnc[2 * s + 0]);
  const float lo_v = ordered_to_float(acc.oMinEnc[2 * s + 1]);
  const float hi_v = ordered_to_float(acc.oMaxEnc[2 * s + 1]);
  r[kColYaw] = yaw;
  r[kColOrientedU] = fmaxf(hi_u - lo_u, 0.f);
  r[kColOrientedV] = fmaxf(hi_v - lo_v, 0.f);
  r[kColOrientedUp] = r[kColMinX + up_axis] <= r[kColMaxX + up_axis]
                          ? (r[kColMaxX + up_axis] - r[kColMinX + up_axis]) : 0.f;
  // Centre of the oriented box, rotated back into world coordinates. Note this is the box CENTRE, not
  // the centroid: the centroid is where the mass is, the centre is the middle of the extents, and for
  // a partially observed object they differ.
  const float mid_u = 0.5f * (lo_u + hi_u);
  const float mid_v = 0.5f * (lo_v + hi_v);
  const float c = cosf(yaw), sn = sinf(yaw);
  float centre[3];
  centre[ua] = c * mid_u - sn * mid_v;
  centre[va] = sn * mid_u + c * mid_v;
  centre[up_axis] = 0.5f * (r[kColMinX + up_axis] + r[kColMaxX + up_axis]);
  r[kColOrientedCenterX + 0] = centre[0];
  r[kColOrientedCenterX + 1] = centre[1];
  r[kColOrientedCenterX + 2] = centre[2];

  row_labels[slot] = static_cast<uint16_t>(s);
}

}  // namespace

void launch_component_filter(const float* positions, const uint16_t* labels,
                             int height, int width, float max_gap_m, float min_fraction,
                             const ComponentBuffers& bufs, cudaStream_t stream) {
  const int64_t count = static_cast<int64_t>(height) * static_cast<int64_t>(width);
  if (count <= 0) return;
  const int64_t grid = grid_for(count);
  const float max_gap_sq = max_gap_m * max_gap_m;

  cudaMemsetAsync(bufs.best, 0, static_cast<size_t>(kInstanceSlots) * sizeof(unsigned long long),
                  stream);
  component_init_kernel<<<grid, kBlock, 0, stream>>>(positions, labels, count, bufs.comp,
                                                     bufs.compCount);
  // No convergence test, hence no host round trip: pointer jumping shortens chains geometrically,
  // so the fixed cap converges with room to spare. Under-convergence would only split a component
  // further, never merge two -- a conservative failure.
  for (int r = 0; r < kComponentRounds; ++r) {
    component_propagate_kernel<<<grid, kBlock, 0, stream>>>(positions, labels, height, width,
                                                            max_gap_sq, bufs.comp);
    component_compress_kernel<<<grid, kBlock, 0, stream>>>(bufs.comp, count);
    component_compress_kernel<<<grid, kBlock, 0, stream>>>(bufs.comp, count);
  }
  component_count_kernel<<<grid, kBlock, 0, stream>>>(bufs.comp, count, bufs.compCount);
  component_best_kernel<<<grid, kBlock, 0, stream>>>(labels, bufs.comp, bufs.compCount, count,
                                                     bufs.best);
  component_select_kernel<<<grid, kBlock, 0, stream>>>(labels, bufs.comp, bufs.compCount, bufs.best,
                                                        count, min_fraction, bufs.labelsOut);
}

void launch_instance_reset(const InstanceAccumulators& acc, cudaStream_t stream) {
  const size_t slots = static_cast<size_t>(kInstanceSlots);

  // Everything that resets to plain zero goes through cudaMemsetAsync, which is a coalesced/DMA
  // fill. The previous version was one thread per SLOT, each looping over its own 3x64 histogram
  // bins -- so adjacent threads wrote 768 bytes apart and the 50 MB histogram fill ran at ~46 GB/s,
  // roughly a fifteenth of the card. That made zeroing the accumulators the THIRD most expensive
  // kernel in the whole application: 1101 us median, 2.56 s over a 103 s profile, 5.6% of GPU 0
  // (nsys, 2026-08-16). It is pure zero-fill; none of that time bought anything.
  //
  // Return codes are deliberately not checked here, matching the kernel launches below: an
  // asynchronous failure surfaces at the `cudaStreamSynchronize` in
  // TcnInstanceStatsOp::compute(), which IS checked and names this operator.
  cudaMemsetAsync(acc.count1, 0, slots * sizeof(uint32_t), stream);
  cudaMemsetAsync(acc.count2, 0, slots * sizeof(uint32_t), stream);
  cudaMemsetAsync(acc.sumUU, 0, slots * sizeof(float), stream);
  cudaMemsetAsync(acc.sumVV, 0, slots * sizeof(float), stream);
  cudaMemsetAsync(acc.sumUV, 0, slots * sizeof(float), stream);
  cudaMemsetAsync(acc.yaw, 0, slots * sizeof(float), stream);
  cudaMemsetAsync(acc.sum1, 0, slots * 3 * sizeof(float), stream);
  cudaMemsetAsync(acc.sqsum1, 0, slots * 3 * sizeof(float), stream);
  cudaMemsetAsync(acc.sigmaRaw, 0, slots * 3 * sizeof(float), stream);
  cudaMemsetAsync(acc.loBound, 0, slots * 3 * sizeof(float), stream);
  cudaMemsetAsync(acc.hiBound, 0, slots * 3 * sizeof(float), stream);
  cudaMemsetAsync(acc.sum2, 0, slots * 3 * sizeof(float), stream);
  cudaMemsetAsync(acc.hist, 0, slots * 3 * kInstanceHistBins * sizeof(uint32_t), stream);

  // 0.f is all-zero bytes, so the float arrays above are byte-fillable. The min/max sentinels are
  // not, so they keep a kernel -- but a flat-indexed, fully coalesced one over ~1.5 MB.
  reset_sentinels_kernel<<<grid_for(kInstanceSlots * 3), kBlock, 0, stream>>>(acc);
}

void launch_instance_pass1(const float* positions, const uint16_t* labels, int64_t count,
                           const InstanceAccumulators& acc, cudaStream_t stream) {
  if (count <= 0) return;
  pass1_kernel<<<grid_for(count), kBlock, 0, stream>>>(positions, labels, count, acc);
}

void launch_instance_finalize_raw(const InstanceAccumulators& acc, cudaStream_t stream) {
  finalize_raw_kernel<<<grid_for(kInstanceSlots), kBlock, 0, stream>>>(acc);
}

void launch_instance_histogram(const float* positions, const uint16_t* labels, int64_t count,
                               const InstanceAccumulators& acc, cudaStream_t stream) {
  if (count <= 0) return;
  histogram_kernel<<<grid_for(count), kBlock, 0, stream>>>(positions, labels, count, acc);
}

void launch_instance_bounds(const InstanceAccumulators& acc, float trim_percentile, float margin,
                            float min_range, cudaStream_t stream) {
  bounds_kernel<<<grid_for(kInstanceSlots), kBlock, 0, stream>>>(acc, trim_percentile, margin,
                                                                min_range);
}

void launch_instance_trimmed(const float* positions, const uint16_t* labels, int64_t count,
                             int up_axis, const InstanceAccumulators& acc, cudaStream_t stream) {
  if (count <= 0) return;
  trimmed_kernel<<<grid_for(count), kBlock, 0, stream>>>(positions, labels, count, up_axis, acc);
}

void launch_instance_yaw(const InstanceAccumulators& acc, int up_axis, float min_anisotropy,
                         cudaStream_t stream) {
  yaw_kernel<<<grid_for(kInstanceSlots), kBlock, 0, stream>>>(acc, up_axis, min_anisotropy);
}

void launch_instance_oriented(const float* positions, const uint16_t* labels, int64_t count,
                              int up_axis, const InstanceAccumulators& acc, cudaStream_t stream) {
  if (count <= 0) return;
  oriented_kernel<<<grid_for(count), kBlock, 0, stream>>>(positions, labels, count, up_axis, acc);
}

void launch_instance_compact(const InstanceAccumulators& acc, int up_axis, int camera_index,
                             uint32_t min_points, float* rows, uint16_t* row_labels,
                             uint32_t* row_count, uint32_t max_rows, cudaStream_t stream) {
  compact_kernel<<<grid_for(kInstanceSlots), kBlock, 0, stream>>>(
      acc, up_axis, camera_index, min_points, rows, row_labels, row_count, max_rows);
}
