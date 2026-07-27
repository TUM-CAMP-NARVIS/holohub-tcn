#pragma once

// camera_math.slang — Shared camera intrinsics, projection parameters, and
// forward Brown-Conrady projection used by both depth_unproject.slang and
// pointcloud_surfels.slang.

// ============================================================================
// Structured camera parameters
// ============================================================================

struct CameraIntrinsics
{
    float fx, fy;               // focal length (pixels)
    float cx, cy;               // principal point (pixels)
    float k1, k2, k3;          // radial distortion (numerator)
    float k4, k5, k6;          // radial distortion (denominator)
    float p1, p2;              // tangential distortion
    float max_radius;          // max normalized radius for valid projection (~1.7)
};

struct ColorProjectionParams
{
    CameraIntrinsics intrinsics;
    uint width;
    uint height;
    float4x4 depth_to_color;   // rigid transform: depth camera -> color camera space
};

// ============================================================================
// Forward projection through full Brown-Conrady distortion model (no Jacobian).
// Faster variant used for color UV computation and surfel splatting.
// ============================================================================

struct ForwardProjectResult
{
    float2 uv;
    bool valid;
};

ForwardProjectResult project_forward(CameraIntrinsics cam, float2 xy)
{
    ForwardProjectResult result;
    result.valid = true;

    float xp = xy.x;
    float yp = xy.y;

    float xp2 = xp * xp;
    float yp2 = yp * yp;
    float xyp = xp * yp;
    float rs = xp2 + yp2;

    if (rs > cam.max_radius * cam.max_radius)
    {
        result.valid = false;
        result.uv = float2(0.f, 0.f);
        return result;
    }

    float rss = rs * rs;
    float rsc = rss * rs;
    float a = 1.f + cam.k1 * rs + cam.k2 * rss + cam.k3 * rsc;
    float b = 1.f + cam.k4 * rs + cam.k5 * rss + cam.k6 * rsc;
    float bi = (b != 0.f) ? (1.f / b) : 1.f;
    float d = a * bi;

    float xp_d = xp * d;
    float yp_d = yp * d;

    float rs_2xp2 = rs + 2.f * xp2;
    float rs_2yp2 = rs + 2.f * yp2;

    xp_d += rs_2xp2 * cam.p2 + 2.f * xyp * cam.p1;
    yp_d += rs_2yp2 * cam.p1 + 2.f * xyp * cam.p2;

    result.uv = float2(xp_d * cam.fx + cam.cx, yp_d * cam.fy + cam.cy);
    return result;
}
