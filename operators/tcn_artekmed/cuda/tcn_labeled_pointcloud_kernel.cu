/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 TUM CAMP / NARVIS. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda_runtime.h>
#include <cmath>

#include "tcn_labeled_pointcloud_kernel.cuh"

namespace {

__global__ void select_class_kernel(const uint16_t* __restrict__ labels,
                                    int64_t count,
                                    int cls,
                                    uint8_t* __restrict__ selected) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const uint16_t label = labels[i];
  // 0 is background in the packed encoding, and a depth pixel with no colour correspondence also
  // carries it (tcn_label_sampler's unlabeled_value), so neither becomes a point.
  const bool hit = label != 0 && (cls < 0 || (label >> kPanopticClassShift) == cls);
  selected[i] = hit ? 1 : 0;
}

__global__ void gather_points_kernel(const float* __restrict__ positions,
                                     const uint16_t* __restrict__ labels,
                                     const int32_t* __restrict__ indices,
                                     int64_t n,
                                     float* __restrict__ out_positions,
                                     uint16_t* __restrict__ out_labels) {
  const int64_t j = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (j >= n) return;
  const int64_t src = indices[j];
  out_positions[3 * j + 0] = positions[3 * src + 0];
  out_positions[3 * j + 1] = positions[3 * src + 1];
  out_positions[3 * j + 2] = positions[3 * src + 2];
  out_labels[j] = labels[src];
}

__global__ void write_empty_point_kernel(float* __restrict__ out_positions,
                                         uint16_t* __restrict__ out_labels) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  const float nan_value = nanf("");
  out_positions[0] = nan_value;
  out_positions[1] = nan_value;
  out_positions[2] = nan_value;
  out_labels[0] = 0;
}

constexpr int kBlock = 256;

int64_t grid_for(int64_t n) {
  return (n + kBlock - 1) / kBlock;
}

}  // namespace

void launch_select_class(const uint16_t* labels,
                         int64_t count,
                         int cls,
                         uint8_t* selected,
                         cudaStream_t stream) {
  if (count <= 0) return;
  select_class_kernel<<<grid_for(count), kBlock, 0, stream>>>(labels, count, cls, selected);
}

void launch_gather_points(const float* positions,
                          const uint16_t* labels,
                          const int32_t* indices,
                          int64_t n,
                          float* out_positions,
                          uint16_t* out_labels,
                          cudaStream_t stream) {
  if (n <= 0) return;
  gather_points_kernel<<<grid_for(n), kBlock, 0, stream>>>(positions, labels, indices, n,
                                                           out_positions, out_labels);
}

void launch_write_empty_point(float* out_positions, uint16_t* out_labels, cudaStream_t stream) {
  write_empty_point_kernel<<<1, 1, 0, stream>>>(out_positions, out_labels);
}
