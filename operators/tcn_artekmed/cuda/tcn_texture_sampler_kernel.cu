// WeightsKernel.cu
#include <cuda_runtime.h>
#include <cmath>
#include "tcn_texture_sampler_kernel.cuh"

// Helper: clamp a value to [lo, hi]
__device__ __forceinline__
float clampf(float x, float lo, float hi)
{
    return fminf(fmaxf(x, lo), hi);
}

// Read one RGBA pixel (uint8) and convert to float4 in [0,1]
__device__ __forceinline__
float4 load_rgba_as_float4(const uint8_t* image, int width, int x, int y)
{
  int idx = (y * width + x) * 4;
  float r = static_cast<float>(image[idx  ]) * (1.0f / 255.0f);
  float g = static_cast<float>(image[idx+1]) * (1.0f / 255.0f);
  float b = static_cast<float>(image[idx+2]) * (1.0f / 255.0f);
  float a = static_cast<float>(image[idx+3]) * (1.0f / 255.0f);
  return make_float4(r, g, b, a);
}


// Bilinear sampler for an HxWx4 uint8 buffer.
// UV in [0,1], boundary-clamped.
__device__ __forceinline__
float4 sample_bilinear_rgba(const uint8_t* image,
                           int width,
                           int height,
                           float u,
                           float v)
{
    // Clamp UV to [0,1] to avoid out-of-bounds
    u = clampf(u, 0.0f, 1.0f);
    v = clampf(v, 0.0f, 1.0f);

    // Map UV to continuous pixel coordinates
    // [0,1] -> [0, width-1] and [0, height-1]
    float x = u * static_cast<float>(width  - 1);
    float y = v * static_cast<float>(height - 1);

    // Integer pixel coordinates
    int x0 = static_cast<int>(floorf(x));
    int y0 = static_cast<int>(floorf(y));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // Clamp neighbors to valid range (handles boundaries)
    x0 = max(0, min(x0, width  - 1));
    x1 = max(0, min(x1, width  - 1));
    y0 = max(0, min(y0, height - 1));
    y1 = max(0, min(y1, height - 1));

    // Fractional part for interpolation weights
    float tx = x - static_cast<float>(x0);
    float ty = y - static_cast<float>(y0);
    float om_tx = 1.f - tx;
    float om_ty = 1.f - ty;

    // Fetch four neighbors
    float4 c00 = load_rgba_as_float4(image, width, x0, y0);
    float4 c10 = load_rgba_as_float4(image, width, x1, y0);
    float4 c01 = load_rgba_as_float4(image, width, x0, y1);
    float4 c11 = load_rgba_as_float4(image, width, x1, y1);

    // Bilinear interpolation
    float4 c0 = make_float4(
        (c00.x * om_tx) + (c10.x * tx),
        (c00.y * om_tx) + (c10.y * tx),
        (c00.z * om_tx) + (c10.z * tx),
        (c00.w * om_tx) + (c10.w * tx)
    );
    float4 c1 = make_float4(
        (c01.x * om_tx) + (c11.x * tx),
        (c01.y * om_tx) + (c11.y * tx),
        (c01.z * om_tx) + (c11.z * tx),
        (c01.w * om_tx) + (c11.w * tx)
    );
    float4 c = make_float4(
        (c0.x * om_ty) + (c1.x * ty),
        (c0.y * om_ty) + (c1.y * ty),
        (c0.z * om_ty) + (c1.z * ty),
        (c0.w * om_ty) + (c1.w * ty)
    );

    return c;
}


__global__ void texture_sampler_rgba_kernel(TextureSamplerParams bp) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= bp.width || y >= bp.height) return;
  const int idx = y * bp.width + x;

  const auto& [u, v] = bp.uv[idx];
  float4 pixel_color = sample_bilinear_rgba(bp.color, bp.colorWidth, bp.colorHeight, u, v);
  //float4 pixel_color = load_rgba_as_float4(bp.color, bp.colorWidth, x, y);
  bp.colorOutput[idx] = make_uchar4(
    static_cast<uint8_t>(pixel_color.x * 255.0f),
    static_cast<uint8_t>(pixel_color.y * 255.0f),
    static_cast<uint8_t>(pixel_color.z * 255.0f),
    static_cast<uint8_t>(pixel_color.w * 255.0f)
  );
}
