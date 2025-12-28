// BackProjectionKernel.cuh
#pragma once

#include "../common/datatypes.hpp"
#include "../common/processing_algorithms.cuh"

struct BackProjectionParams {
  const uint16_t* depth;    // [H*W]
  const float2*   xy;       // [H*W], per-pixel (x,y)
  float* positions;         // [H*W*3]
  float* texcoords;         // [H*W*2]
  float* depth_float;         // [H*W]
  int width;
  int height;
  float depth_units_per_meter;
  float near_limit_m;
  float far_limit_m;
  CameraParameters color_params;
  float4x4 color_to_depth;   // maps depth->color space
  float4x4 depth_extrinsics; // maps depth->world (or desired output space)
  bool positions_enabled;
  bool texcoords_enabled;
  bool depth_float_enabled;
};

__global__ void backprojection_u16_kernel(BackProjectionParams bp);