#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <Eigen/Core>

// Simple math structs compatible with CUDA device code
struct float2 { float x, y; };
struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };

struct float4x4 {
  // Column-major like in original code
  float4 col[4];
};

// Vertex structure (kept for parity, used internally)
struct VertexPositionTexcoord {
  float3 position;
  float2 texcoord;
};

// Camera model parameters (Brown–Conrady as used in the original)
struct CameraParameters {
  float fx{0}, fy{0}, cx{0}, cy{0};
  float k1{0}, k2{0}, k3{0}, k4{0}, k5{0}, k6{0};
  float codx{0}, cody{0};
  float p1{0}, p2{0};
  int   is_distorted{0}; // 1 = distorted, 0 = undistorted
};

// Helper to cast Eigen::Matrix4f (row-major) to our float4x4 (column array)
inline float4x4 float4x4Cast(const Eigen::Matrix4f& m) {
  float4x4 out{};
  // Fill columns
  for (int c = 0; c < 4; ++c) {
    out.col[c].x = m(0, c);
    out.col[c].y = m(1, c);
    out.col[c].z = m(2, c);
    out.col[c].w = m(3, c);
  }
  return out;
}