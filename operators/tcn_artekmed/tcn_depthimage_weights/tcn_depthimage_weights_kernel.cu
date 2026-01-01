// WeightsKernel.cu
#include <cuda_runtime.h>
#include <cmath>
#include "tcn_depthimage_weights_kernel.cuh"
#include "../common/cuda_util_math.h"
#include "../common/processing_algorithms.cuh"

__global__ void compute_weights_u16_kernel(WeightsParams bp) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= bp.width || y >= bp.height)
    return;

  const int idx = y * bp.width + x;

  const int north_idx = (y-1) * bp.width + x;
  const int south_idx = (y+1) * bp.width + x;
  const int west_idx = y * bp.width + (x-1);
  const int east_idx = y * bp.width + (x+1);

  if (x < 0 || x >= bp.width || y < 0 || y >= bp.height) {
    bp.computeWeightsOutput[idx] = 0.0f;
  } else {
    float outputWeight = 0.f;

    const uint16_t depth = bp.depth[idx];
    const float2& xy = bp.xy[idx];
    const float depthf = (float)depth / bp.depthUnitsPerMeter;

    if (isValidDepth(depthf)) {
      float3 centerPos;
      centerPos.x = xy.x * depthf;
      centerPos.y = xy.y * depthf;
      centerPos.z = depthf;

      // @todo: this method of computing the normal is super trivial - maybe needed for speed
      //        but we could consider using a more elaborate method that takes multiple neighbours
      //        into account
      const float north = y != 0 ? (float)bp.depth[north_idx] / bp.depthUnitsPerMeter : 0.f;
      const float south = y != bp.width - 1 ? (float)bp.depth[south_idx] / bp.depthUnitsPerMeter : 0.f;
      const float west = x != 0 ? (float)bp.depth[west_idx] / bp.depthUnitsPerMeter : 0.f;
      const float east = x != bp.height - 1 ? (float)bp.depth[east_idx] / bp.depthUnitsPerMeter : 0.f;

      float3 tangU{}, tangV{};
      bool valid_neighbors{true};
      if (isValidDepth(north) && y > 0) {
        const float2 xyV = bp.xy[north_idx];
        tangV.x = xyV.x * north;
        tangV.y = xyV.y * north;
        tangV.z = north;
      } else if (isValidDepth(south)) {
        const float2 xyV = bp.xy[south_idx];
        tangV.x = xyV.x * south;
        tangV.y = xyV.y * south;
        tangV.z = south;
      } else {
        valid_neighbors = false;
      }

      if (isValidDepth(west) && x > 0) {
        const float2 xyU = bp.xy[west_idx];
        tangU.x = xyU.x * west;
        tangU.y = xyU.y * west;
        tangU.z = west;
      } else if (isValidDepth(east)) {
        const float2 xyU = bp.xy[east_idx];
        tangU.x = xyU.x * east;
        tangU.y = xyU.y * east;
        tangU.z = east;
      } else {
        valid_neighbors = false;
      }

      if (valid_neighbors) {
        float3 normal = normalize(cross(tangU - centerPos, tangV - centerPos));

        align_normal(normal, centerPos, {0.f, 0.f, 0.f});

        outputWeight = cameraQualityWeight(centerPos,
                                           {x, y},
                                           depthf,
                                           normal,
                                           bp.height,
                                           bp.width,
                                           {0, 0, 0},
                                           bp.quality_weight_params);
      }
    }
    bp.computeWeightsOutput[idx] = outputWeight;
  }
}