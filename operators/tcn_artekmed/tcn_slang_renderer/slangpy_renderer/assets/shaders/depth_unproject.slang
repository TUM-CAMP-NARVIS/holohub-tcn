// depth_unproject.slang — Compute kernels for GPU depth-to-pointcloud conversion
// with optional projection into a color camera's image plane.
//
// 1. compute_xy_table: Builds an XY lookup table by iteratively un-distorting each pixel
//    (Newton-Raphson solver, Brown-Conrady distortion model). Dispatched once per camera.
// 2. compute_pointcloud: Multiplies each pixel's XY ray by its depth value, and optionally
//    projects each 3D point into the color camera to produce normalized UV texcoords.
//    Dispatched per frame.
//
// Ported from: https://github.com/TUM-CAMP-NARVIS/xylt/blob/master/lib/library.cpp
// Color projection ported from: tcn_depthimage_backprojection_kernel.cu

#include "camera_math.slang"

// ============================================================================
// Depth-specific parameters (not shared)
// ============================================================================

struct DepthParameters
{
    uint width;
    uint height;
    CameraIntrinsics intrinsics;
    float depth_scale;         // raw uint16 -> meters (0.001 for 1mm)
};

uniform DepthParameters depth_params;
uniform ColorProjectionParams color_params;
uniform bool compute_texcoords;

// ============================================================================
// Buffers
// ============================================================================

uniform RWStructuredBuffer<float2> xy_table;          // XY table output (compute_xy_table)
uniform StructuredBuffer<float2> xy_table_in;         // XY table input (compute_pointcloud)
uniform Texture2D<uint> depth_texture;                // depth image input (r16_uint)
uniform RWStructuredBuffer<float3> pointcloud;        // pointcloud output
uniform RWStructuredBuffer<float2> texcoords;         // texcoord output (normalized UVs)
uniform RWStructuredBuffer<float3> normals;            // normal output (compute_normals)

// ============================================================================
// Forward projection through full Brown-Conrady distortion model.
// Returns the projected pixel coordinate and the 2x2 Jacobian.
// Used by the Newton-Raphson un-projection solver.
// ============================================================================

struct ProjectResult
{
    float2 uv;       // projected pixel coordinate
    float4 J;        // Jacobian: [du/dxp, du/dyp, dv/dxp, dv/dyp]
    bool valid;
};

ProjectResult project_internal(CameraIntrinsics cam, float2 xy)
{
    ProjectResult result;
    result.valid = true;

    // Center of distortion is 0 for Brown-Conrady model
    float xp = xy.x;
    float yp = xy.y;

    float xp2 = xp * xp;
    float yp2 = yp * yp;
    float xyp = xp * yp;
    float rs = xp2 + yp2;

    if (rs > cam.max_radius * cam.max_radius)
    {
        result.valid = false;
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

    result.uv.x = xp_d * cam.fx + cam.cx;
    result.uv.y = yp_d * cam.fy + cam.cy;

    // Compute Jacobian
    float dudrs = cam.k1 + 2.f * cam.k2 * rs + 3.f * cam.k3 * rss;
    float dvdrs = cam.k4 + 2.f * cam.k5 * rs + 3.f * cam.k6 * rss;
    float bis = bi * bi;
    float dddrs = (dudrs * b - a * dvdrs) * bis;

    float dddrs_2 = dddrs * 2.f;
    float xp_dddrs_2 = xp * dddrs_2;
    float yp_xp_dddrs_2 = yp * xp_dddrs_2;

    result.J.x = cam.fx * (d + xp * xp_dddrs_2 + 6.f * xp * cam.p2 + 2.f * yp * cam.p1);
    result.J.y = cam.fx * (yp_xp_dddrs_2 + 2.f * yp * cam.p2 + 2.f * xp * cam.p1);
    result.J.z = cam.fy * (yp_xp_dddrs_2 + 2.f * xp * cam.p1 + 2.f * yp * cam.p2);
    result.J.w = cam.fy * (d + yp * yp * dddrs_2 + 6.f * yp * cam.p1 + 2.f * xp * cam.p2);

    return result;
}

// ============================================================================
// Invert a 2x2 matrix stored as float4 [a, b, c, d] -> [[a,b],[c,d]]
// ============================================================================

float4 invert_2x2(float4 J)
{
    float det = J.x * J.w - J.y * J.z;
    float inv_det = 1.f / det;
    return float4(inv_det * J.w, -inv_det * J.y, -inv_det * J.z, inv_det * J.x);
}

// ============================================================================
// Iterative Newton-Raphson un-projection.
// Given target pixel coordinate `uv`, refine `xy` to minimize re-projection error.
// ============================================================================

struct UnprojectResult
{
    float2 xy;
    bool valid;
};

UnprojectResult iterative_unproject(CameraIntrinsics cam, float2 uv, float2 xy, uint max_passes)
{
    UnprojectResult result;
    result.valid = true;

    float2 best_xy = float2(0.f, 0.f);
    float best_err = 3.402823466e+38f;  // FLT_MAX

    for (uint pass = 0; pass < max_passes; pass++)
    {
        ProjectResult proj = project_internal(cam, xy);
        if (!proj.valid)
        {
            result.valid = false;
            result.xy = xy;
            return result;
        }

        float err_x = uv.x - proj.uv.x;
        float err_y = uv.y - proj.uv.y;
        float err = err_x * err_x + err_y * err_y;

        if (err >= best_err)
        {
            xy = best_xy;
            break;
        }

        best_err = err;
        best_xy = xy;

        float4 Jinv = invert_2x2(proj.J);

        if (pass + 1 == max_passes || best_err < 1e-22f)
        {
            break;
        }

        float dx = Jinv.x * err_x + Jinv.y * err_y;
        float dy = Jinv.z * err_x + Jinv.w * err_y;

        xy.x += dx;
        xy.y += dy;
    }

    if (best_err > 1e-6f)
    {
        result.valid = false;
    }

    result.xy = xy;
    return result;
}

// ============================================================================
// Full un-projection: initial estimate + iterative refinement.
// Returns the normalized ray direction (x, y) at unit depth for a given pixel.
// ============================================================================

UnprojectResult unproject_pixel(CameraIntrinsics cam, float2 uv)
{
    // Initial estimate: invert radial distortion
    float xp_d = (uv.x - cam.cx) / cam.fx;
    float yp_d = (uv.y - cam.cy) / cam.fy;

    float rs = xp_d * xp_d + yp_d * yp_d;
    float rss = rs * rs;
    float rsc = rss * rs;
    float a = 1.f + cam.k1 * rs + cam.k2 * rss + cam.k3 * rsc;
    float b = 1.f + cam.k4 * rs + cam.k5 * rss + cam.k6 * rsc;
    float ai = (a != 0.f) ? (1.f / a) : 1.f;
    float di = ai * b;

    float2 xy;
    xy.x = xp_d * di;
    xy.y = yp_d * di;

    // Approximate tangential correction
    float two_xy = 2.f * xy.x * xy.y;
    float xx = xy.x * xy.x;
    float yy = xy.y * xy.y;

    xy.x -= (yy + 3.f * xx) * cam.p2 + two_xy * cam.p1;
    // NOTE: The original C++ code uses (xx + 3*xx) here, which appears to be a bug
    // (should likely be xx + 3*yy for standard Brown-Conrady). Ported faithfully.
    xy.y -= (xx + 3.f * xx) * cam.p1 + two_xy * cam.p2;

    // Iterative Newton-Raphson refinement (up to 20 passes)
    return iterative_unproject(cam, uv, xy, 20);
}

// ============================================================================
// Kernel 1: Compute XY Lookup Table
// Dispatched once per camera setup. For each pixel, computes the normalized
// ray direction at unit depth.
// ============================================================================

[shader("compute")]
[numthreads(16, 16, 1)]
void compute_xy_table(uint3 threadId : SV_DispatchThreadID)
{
    uint px = threadId.x;
    uint py = threadId.y;

    if (px >= depth_params.width || py >= depth_params.height)
        return;

    uint idx = py * depth_params.width + px;
    float2 pixel = float2((float)px, (float)py);

    UnprojectResult result = unproject_pixel(depth_params.intrinsics, pixel);

    if (result.valid)
    {
        xy_table[idx] = result.xy;
    }
    else
    {
        xy_table[idx] = float2(0.f, 0.f);
    }
}

// ============================================================================
// Kernel 2: Compute Pointcloud (+ optional color UVs) from Depth + XY Table
// Dispatched every frame. Reads depth, multiplies by precomputed ray direction.
// When compute_texcoords is true, also transforms each 3D point into the color
// camera's coordinate frame and projects through its distortion model to produce
// normalized UV texture coordinates.
// ============================================================================

[shader("compute")]
[numthreads(16, 16, 1)]
void compute_pointcloud(uint3 threadId : SV_DispatchThreadID)
{
    uint px = threadId.x;
    uint py = threadId.y;

    if (px >= depth_params.width || py >= depth_params.height)
        return;

    uint idx = py * depth_params.width + px;

    uint depth_raw = depth_texture.Load(int3(px, py, 0));

    if (depth_raw == 0)
    {
        pointcloud[idx] = float3(0.f, 0.f, 0.f);
        texcoords[idx] = float2(0.f, 0.f);
        return;
    }

    float depth_m = (float)depth_raw * depth_params.depth_scale;
    float2 ray = xy_table_in[idx];

    float3 p_depth = float3(ray.x * depth_m, ray.y * depth_m, depth_m);
    pointcloud[idx] = p_depth;

    if (compute_texcoords)
    {
        // Transform from depth camera space to color camera space
        float4 p_color4 = mul(color_params.depth_to_color, float4(p_depth, 1.0));
        float3 p_color = p_color4.xyz;

        // Perspective divide to get normalized image coordinates
        float2 xy_norm = float2(p_color.x / p_color.z, p_color.y / p_color.z);

        // Forward projection through color camera distortion model
        ForwardProjectResult proj = project_forward(color_params.intrinsics, xy_norm);

        if (proj.valid)
        {
            float2 uv = proj.uv / float2((float)color_params.width, (float)color_params.height);

            // Invalidate points that project outside the color image.
            // This happens when the depth camera's FoV is wider than the color
            // camera's — edge pixels have no corresponding color data.
            if (uv.x < 0.f || uv.x > 1.f || uv.y < 0.f || uv.y > 1.f)
            {
                pointcloud[idx] = float3(0.f, 0.f, 0.f);
                texcoords[idx] = float2(0.f, 0.f);
            }
            else
            {
                texcoords[idx] = uv;
            }
        }
        else
        {
            // Projection failed (outside distortion model radius).
            // Invalidate the point entirely.
            pointcloud[idx] = float3(0.f, 0.f, 0.f);
            texcoords[idx] = float2(0.f, 0.f);
        }
    }
    else
    {
        texcoords[idx] = float2(0.f, 0.f);
    }
}

// ============================================================================
// Kernel 3: Compute Normals from Structured Pointcloud Grid
// Dispatched after compute_pointcloud. Computes per-pixel surface normals via
// finite differences on the 3D position grid: cross(dx, dy) where dx and dy
// are central differences (with one-sided fallback at boundaries/invalid pixels).
// Convention: cross(right, down) produces +z for fronto-parallel surfaces.
// ============================================================================

[shader("compute")]
[numthreads(16, 16, 1)]
void compute_normals(uint3 threadId : SV_DispatchThreadID)
{
    uint px = threadId.x;
    uint py = threadId.y;

    if (px >= depth_params.width || py >= depth_params.height)
        return;

    uint idx = py * depth_params.width + px;
    float3 center = pointcloud[idx];

    // Invalid point — write zero normal
    if (center.z == 0.f)
    {
        normals[idx] = float3(0.f, 0.f, 0.f);
        return;
    }

    // --- Horizontal finite difference (dx) ---
    bool has_left = false;
    bool has_right = false;
    float3 left_pt, right_pt;

    if (px > 0)
    {
        left_pt = pointcloud[idx - 1];
        has_left = (left_pt.z != 0.f);
    }
    if (px + 1 < depth_params.width)
    {
        right_pt = pointcloud[idx + 1];
        has_right = (right_pt.z != 0.f);
    }

    float3 dx;
    bool dx_valid = false;
    if (has_left && has_right)
    {
        dx = (right_pt - left_pt) * 0.5f;
        dx_valid = true;
    }
    else if (has_right)
    {
        dx = right_pt - center;
        dx_valid = true;
    }
    else if (has_left)
    {
        dx = center - left_pt;
        dx_valid = true;
    }

    // --- Vertical finite difference (dy) ---
    bool has_up = false;
    bool has_down = false;
    float3 up_pt, down_pt;

    if (py > 0)
    {
        up_pt = pointcloud[idx - depth_params.width];
        has_up = (up_pt.z != 0.f);
    }
    if (py + 1 < depth_params.height)
    {
        down_pt = pointcloud[idx + depth_params.width];
        has_down = (down_pt.z != 0.f);
    }

    float3 dy;
    bool dy_valid = false;
    if (has_up && has_down)
    {
        dy = (down_pt - up_pt) * 0.5f;
        dy_valid = true;
    }
    else if (has_down)
    {
        dy = down_pt - center;
        dy_valid = true;
    }
    else if (has_up)
    {
        dy = center - up_pt;
        dy_valid = true;
    }

    if (!dx_valid || !dy_valid)
    {
        normals[idx] = float3(0.f, 0.f, 0.f);
        return;
    }

    // cross(right, down) => outward normal (+z for fronto-parallel surfaces)
    float3 n = cross(dx, dy);
    float len = length(n);
    if (len > 0.f)
    {
        normals[idx] = n / len;
    }
    else
    {
        normals[idx] = float3(0.f, 0.f, 0.f);
    }
}
