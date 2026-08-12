/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 TUM CAMP / NARVIS. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Per-instance reduction over a labeled point grid: count, centroid, axis-aligned box and per-axis
 * spread for every packed panoptic label present.
 *
 * This is a reduction BY KEY, not a clustering problem: the masks have already segmented the points,
 * so every point carries the instance it belongs to. The only clustering-shaped concern left is a
 * single mask covering more than one physical surface, which the sigma trim mitigates -- see the
 * operator's README for where that is and is not sufficient.
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

__global__ void reset_kernel(InstanceAccumulators acc) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= kInstanceSlots) return;
  acc.count1[s] = 0u;
  acc.count2[s] = 0u;
  for (int d = 0; d < 3; ++d) {
    const int k = 3 * s + d;
    acc.sum1[k] = 0.f;
    acc.sqsum1[k] = 0.f;
    acc.mean[k] = 0.f;
    acc.sigma[k] = 0.f;
    acc.sum2[k] = 0.f;
    acc.minEnc[k] = 0x7FFFFFFF;      // +inf under the ordered encoding
    acc.maxEnc[k] = static_cast<int32_t>(0x80000000);   // -inf
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
  const float x = positions[3 * i + 0];
  const float y = positions[3 * i + 1];
  const float z = positions[3 * i + 2];
  // Defensive: a labeled pixel should always have valid depth (an invalid sample yields a NaN
  // texcoord, hence no label), but a non-finite position would poison every aggregate it reaches.
  if (!isfinite(x) || !isfinite(y) || !isfinite(z)) return;

  atomicAdd(&acc.count1[label], 1u);
  atomicAdd(&acc.sum1[3 * label + 0], x);
  atomicAdd(&acc.sum1[3 * label + 1], y);
  atomicAdd(&acc.sum1[3 * label + 2], z);
  atomicAdd(&acc.sqsum1[3 * label + 0], x * x);
  atomicAdd(&acc.sqsum1[3 * label + 1], y * y);
  atomicAdd(&acc.sqsum1[3 * label + 2], z * z);
}

__global__ void finalize1_kernel(InstanceAccumulators acc) {
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
    acc.mean[k] = mean;
    acc.sigma[k] = sqrtf(var);
  }
}

__global__ void pass2_kernel(const float* __restrict__ positions,
                             const uint16_t* __restrict__ labels,
                             int64_t count,
                             float sigma_k,
                             float sigma_floor,
                             InstanceAccumulators acc) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const uint16_t label = labels[i];
  if (label == 0) return;
  const float p[3] = {positions[3 * i + 0], positions[3 * i + 1], positions[3 * i + 2]};
  if (!isfinite(p[0]) || !isfinite(p[1]) || !isfinite(p[2])) return;

  // Reject per axis rather than by Euclidean distance: an axis-aligned box is what this produces, so
  // trimming per axis is what actually tightens it.
  for (int d = 0; d < 3; ++d) {
    const float limit = sigma_k * fmaxf(acc.sigma[3 * label + d], sigma_floor);
    if (fabsf(p[d] - acc.mean[3 * label + d]) > limit) return;
  }

  atomicAdd(&acc.count2[label], 1u);
  for (int d = 0; d < 3; ++d) {
    atomicAdd(&acc.sum2[3 * label + d], p[d]);
    atomicMin(&acc.minEnc[3 * label + d], float_to_ordered(p[d]));
    atomicMax(&acc.maxEnc[3 * label + d], float_to_ordered(p[d]));
  }
}

__global__ void compact_kernel(InstanceAccumulators acc,
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
    r[kColSigmaX + d] = acc.sigma[3 * s + d];   // spread BEFORE trimming: the bleed indicator
  }
  row_labels[slot] = static_cast<uint16_t>(s);
}

}  // namespace

void launch_instance_reset(const InstanceAccumulators& acc, cudaStream_t stream) {
  reset_kernel<<<grid_for(kInstanceSlots), kBlock, 0, stream>>>(acc);
}

void launch_instance_pass1(const float* positions, const uint16_t* labels, int64_t count,
                           const InstanceAccumulators& acc, cudaStream_t stream) {
  if (count <= 0) return;
  pass1_kernel<<<grid_for(count), kBlock, 0, stream>>>(positions, labels, count, acc);
}

void launch_instance_finalize1(const InstanceAccumulators& acc, cudaStream_t stream) {
  finalize1_kernel<<<grid_for(kInstanceSlots), kBlock, 0, stream>>>(acc);
}

void launch_instance_pass2(const float* positions, const uint16_t* labels, int64_t count,
                           float sigma_k, float sigma_floor,
                           const InstanceAccumulators& acc, cudaStream_t stream) {
  if (count <= 0) return;
  pass2_kernel<<<grid_for(count), kBlock, 0, stream>>>(positions, labels, count,
                                                      sigma_k, sigma_floor, acc);
}

void launch_instance_compact(const InstanceAccumulators& acc, int camera_index, uint32_t min_points,
                             float* rows, uint16_t* row_labels, uint32_t* row_count,
                             uint32_t max_rows, cudaStream_t stream) {
  compact_kernel<<<grid_for(kInstanceSlots), kBlock, 0, stream>>>(
      acc, camera_index, min_points, rows, row_labels, row_count, max_rows);
}
