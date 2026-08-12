// BackProjectionKernel.cu
#include <cuda_runtime.h>
#include <cmath>
#include "tcn_depthimage_backprojection_kernel.cuh"
#include "detail/processing_algorithms.cuh"

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

  const bool want_positions = bp.positions_enabled && bp.positions;
  const bool want_texcoords = bp.texcoords_enabled && bp.texcoords;

  // Both outputs are driven from the same unprojected point, but neither may gate the other: this
  // block used to be nested inside `want_positions`, so enable_positions=false with
  // enable_texcoords=true emitted an untouched texcoord buffer -- silently, and precisely in the
  // configuration that wants texcoords without a point cloud.
  if (want_positions || want_texcoords) {
    float3 out_pos = {0.f, 0.f, 0.f};
    // NaN, not (0,0), for "this pixel has no colour correspondence". (0,0) is a legitimate
    // texcoord -- the colour image's top-left pixel -- so an invalid-depth pixel was
    // indistinguishable from one that genuinely projects into the corner, and a label sampler
    // would hand it that corner's label.
    //
    // Behaviour-preserving for the colour sampler, which clamps with fminf(fmaxf(u,0),1): IEEE
    // fmaxf(NaN,0) returns 0, so a NaN texcoord clamps to 0 and samples pixel (0,0) exactly as
    // (0,0) did. Only a consumer that checks for NaN sees any difference.
    float2 out_uv  = {nanf(""), nanf("")};

    if (!isnan(depth_m) && !isinf(depth_m) && depth_m >= bp.near_limit_m && depth_m <= bp.far_limit_m) {
      const float2 xy = bp.xy[idx];
      // point in depth camera space (note sign convention)
      float3 p;
      p.x = xy.x * depth_m;
      p.y = -xy.y * depth_m;
      p.z = -depth_m;

      if (want_texcoords) {
        // transform to color camera space and project
        float3 p_color; transform_point_matrix(p_color, p, bp.depth_to_color);
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
    if (want_positions) {
      bp.positions[3 * idx + 0] = out_pos.x;
      bp.positions[3 * idx + 1] = out_pos.y;
      bp.positions[3 * idx + 2] = out_pos.z;
    }

    if (want_texcoords) {
      bp.texcoords[2 * idx + 0] = out_uv.x / bp.color_width;
      bp.texcoords[2 * idx + 1] = out_uv.y / bp.color_height;
    }
  }

}
