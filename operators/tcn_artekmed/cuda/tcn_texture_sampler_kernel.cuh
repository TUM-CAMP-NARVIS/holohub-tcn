// WeightsKernel.cuh
#pragma once

#include "../common/datatypes.hpp"

struct TextureSamplerParams {
  const uint8_t* color;    // [H*W*4], per-pixel (rgba)
  const float2*   uv;       // [H*W], per-pixel (uv)

  // outputs
  uchar4* colorOutput;

  // general parameters
  int width;
  int height;
  int colorWidth;
  int colorHeight;

};


__global__ void texture_sampler_rgba_kernel(TextureSamplerParams bp);