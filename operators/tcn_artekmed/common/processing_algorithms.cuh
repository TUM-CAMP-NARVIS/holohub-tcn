#pragma once

#include "datatypes.hpp"

// Transform a 3D point by a 4x4 matrix (column-major struct)
__host__ __device__ inline void transform_point_matrix(float3& out_point,
                                                      const float3& in_point,
                                                      const float4x4& extrinsics) {
  const float4 col0 = extrinsics.col[0];
  const float4 col1 = extrinsics.col[1];
  const float4 col2 = extrinsics.col[2];
  const float4 col3 = extrinsics.col[3];
  out_point.x = col0.x * in_point.x + col1.x * in_point.y + col2.x * in_point.z + col3.x;
  out_point.y = col0.y * in_point.x + col1.y * in_point.y + col2.y * in_point.z + col3.y;
  out_point.z = col0.z * in_point.x + col1.z * in_point.y + col2.z * in_point.z + col3.z;
}

__host__ __device__ inline float3 transform_point_matrix(const float3& p,
                                                         const float4x4& extrinsics) {
  float3 out; transform_point_matrix(out, p, extrinsics); return out;
}

// Undistorted pinhole projection with our sign convention
__host__ __device__ inline void project_point_to_image_plane_undistorted(
    float2& uv, const float3& point, const CameraParameters& cam) {
  uv.x = -point.x * cam.fx / point.z + cam.cx;
  uv.y =  point.y * cam.fy / point.z + cam.cy; // y-flip is already in point construction
}

// Brown–Conrady distortion model (exactly as used in the original transformations)
__host__ __device__ inline void project_point_to_image_plane_distorted(
    float2& uv, const float3& point, const CameraParameters& cam) {
  const float cx = cam.cx, cy = cam.cy, fx = cam.fx, fy = cam.fy;
  const float k1 = cam.k1, k2 = cam.k2, k3 = cam.k3, k4 = cam.k4, k5 = cam.k5, k6 = cam.k6;
  const float codx = cam.codx, cody = cam.cody, p1 = cam.p1, p2 = cam.p2;

  float xp = -point.x / point.z - codx;
  float yp =  point.y / point.z - cody;

  const float xp2 = xp * xp;
  const float yp2 = yp * yp;
  const float xyp = xp * yp;
  const float rs  = xp2 + yp2;
  const float rss = rs * rs;
  const float rsc = rss * rs;
  const float a = 1.f + k1 * rs + k2 * rss + k3 * rsc;
  const float b = 1.f + k4 * rs + k5 * rss + k6 * rsc;
  const float bi = (b != 0.f) ? (1.f / b) : 1.f;
  const float d = a * bi;

  float xp_d = xp * d;
  float yp_d = yp * d;
  const float rs_2xp2 = rs + 2.f * xp2;
  const float rs_2yp2 = rs + 2.f * yp2;
  xp_d += rs_2xp2 * p2 + 2.f * xyp * p1;
  yp_d += rs_2yp2 * p1 + 2.f * xyp * p2;

  const float xp_d_cx = xp_d + codx;
  const float yp_d_cy = yp_d + cody;
  uv.x = xp_d_cx * fx + cx;
  uv.y = yp_d_cy * fy + cy;
}