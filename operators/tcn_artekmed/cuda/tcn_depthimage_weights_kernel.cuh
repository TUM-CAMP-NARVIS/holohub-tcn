// WeightsKernel.cuh
#pragma once

#include "../common/datatypes.hpp"

struct WeightsParams {
  const uint16_t* depth;    // [H*W]
  const float2*   xy;       // [H*W], per-pixel (x,y)

  // outputs
  float* computeWeightsOutput;

  // general parameters
  int width;
  int height;

  // compute weights params
  float depthUnitsPerMeter;
  CameraQualityWeightParams quality_weight_params;
};


__global__ void compute_weights_u16_kernel(WeightsParams bp);