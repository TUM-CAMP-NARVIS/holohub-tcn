// BackProjectionKernel.cu
#include <cuda_runtime.h>
#include <cmath>
#include "tcn_depthimage_backprojection_kernel.cuh"

__global__ void backprojection_u16_kernel(BackProjectionParams bp) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= bp.width || y >= bp.height) return;
  const int idx = y * bp.width + x;

  const uint16_t d = bp.depth[idx];
  const float depth_m = static_cast<float>(d) / bp.depth_units_per_meter;

  if (bp.depth_float_enabled && bp.depth_float) {
    bp.depth_float[idx] = depth_m;
  }

  if (bp.positions_enabled && bp.positions) {
    float3 out_pos = {0.f, 0.f, 0.f};
    float2 out_uv  = {0.f, 0.f};

    if (!isnan(depth_m) && !isinf(depth_m) && depth_m >= bp.near_limit_m && depth_m <= bp.far_limit_m) {
      const float2 xy = bp.xy[idx];
      // point in depth camera space (note sign convention)
      float3 p;
      p.x = xy.x * depth_m;
      p.y = -xy.y * depth_m;
      p.z = -depth_m;

      if (bp.texcoords_enabled && bp.texcoords) {
        // transform to color camera space and project
        float3 p_color; transform_point_matrix(p_color, p, bp.color_to_depth);
        if (bp.color_params.is_distorted)
          project_point_to_image_plane_distorted(out_uv, p_color, bp.color_params);
        else
          project_point_to_image_plane_undistorted(out_uv, p_color, bp.color_params);
      }

      // transform to output space (depth_extrinsics)
      out_pos = transform_point_matrix(p, bp.depth_extrinsics);

      // Pixel-space UV is kept (matching original semantics prior to optional normalization)
    }

    // Write planar outputs
    bp.positions[3 * idx + 0] = out_pos.x;
    bp.positions[3 * idx + 1] = out_pos.y;
    bp.positions[3 * idx + 2] = out_pos.z;

    if (bp.texcoords_enabled && bp.texcoords) {
      bp.texcoords[2 * idx + 0] = out_uv.x;
      bp.texcoords[2 * idx + 1] = out_uv.y;
    }
  }

}