// pointcloud_surfels.slang — Normal-oriented textured hexagonal surfel rendering.
//
// Geometry shader expands each point into a hexagonal surfel oriented by the
// point's normal vector. Each surfel vertex gets its UV by projecting the 3D
// position through the color camera model (Brown-Conrady distortion).

import "slangpy";
#include "camera_math.slang"

// --- Uniforms ---
uniform float4x4 view;
uniform float4x4 proj;
uniform float4x4 model;

// Point cloud data buffers
uniform StructuredBuffer<float3> vertices;
uniform StructuredBuffer<float3> normals;
uniform StructuredBuffer<float2> uvCoords;  // center UVs (fallback, from compute shader)

uniform Texture2D<float4> colorTex;
uniform SamplerState sampler_colorTex;

// Color camera parameters for per-vertex UV projection
uniform ColorProjectionParams color_camera;
uniform bool has_color_projection;

// Sprite sizing
uniform float depth_fy;       // depth camera fy (pixels) for auto-sizing
uniform float sprite_scale;   // multiplier on auto-computed radius (default 1.5)

// Depth image dimensions for structured grid indexing
uniform int depthWidth;
uniform int depthHeight;

// Debug: render with static color instead of texture
uniform bool useStaticColor;
uniform float4 staticColor;

// Edge falloff: alpha = 1 - pow(dist, falloff_exponent)
// Higher exponent preserves detail in center, soft blend at edges.
uniform float falloff_exponent;  // default 4.0 (set from Python)

// --- Inter-stage structures ---

struct VertexToGeometry {
    uint vid : VERTEXID;
};

struct GeometryToFragment {
    float2 uv     : TEXCOORD0;
    float  dist   : TEXCOORD1;  // 0 at center, 1 at hex edge
    float4 pos    : SV_Position;
};

// --- Vertex shader: pass-through ---

[shader("vertex")]
VertexToGeometry vertex_main(uint vid : SV_VertexID) {
    VertexToGeometry output;
    output.vid = vid;
    return output;
}

// --- Helper: transform to clip space ---

float4 world_to_clip(float3 world_pos) {
    float4 wp = float4(world_pos, 1.0);
    return mul(proj, mul(view, mul(model, wp)));
}

// --- Helper: project 3D point to color camera UV ---

float2 project_to_color_uv(float3 depth_pos) {
    // Transform from depth camera space to color camera space
    float4 p_color4 = mul(color_camera.depth_to_color, float4(depth_pos, 1.0));
    float3 p_color = p_color4.xyz;

    if (p_color.z <= 0.f)
        return float2(-1.f, -1.f);

    // Perspective divide
    float2 xy_norm = float2(p_color.x / p_color.z, p_color.y / p_color.z);

    // Forward project through distortion model
    ForwardProjectResult proj_result = project_forward(color_camera.intrinsics, xy_norm);

    if (!proj_result.valid)
        return float2(-1.f, -1.f);

    // Normalize to [0, 1]
    return proj_result.uv / float2((float)color_camera.width, (float)color_camera.height);
}

// --- Constants for hexagon ---

static const int HEXAGON_VERTS = 6;
// Precomputed cos/sin for 60-degree increments (0, 60, 120, 180, 240, 300)
static const float2 hex_offsets[6] = {
    float2( 1.0,       0.0),         // 0°
    float2( 0.5,       0.8660254),   // 60°  (sin60 = sqrt(3)/2)
    float2(-0.5,       0.8660254),   // 120°
    float2(-1.0,       0.0),         // 180°
    float2(-0.5,      -0.8660254),   // 240°
    float2( 0.5,      -0.8660254),   // 300°
};

// --- Geometry shader ---

[shader("geometry")]
[maxvertexcount(18)]   // 6 triangles * 3 vertices = 18
void geometry_main(
    point VertexToGeometry input[1],
    inout TriangleStream<GeometryToFragment> outStream)
{
    uint idx = input[0].vid;
    float3 pos = vertices[idx];
    float3 normal = normals[idx];

    // Skip invalid points (zero position or zero normal)
    if (pos.z <= 0.f || length(normal) < 0.5f)
        return;

    // Auto-compute sprite radius from depth and camera intrinsics
    // At depth z, each pixel covers z/fy meters vertically.
    // sprite_scale controls overlap (1.5 = slight overlap to fill gaps).
    float radius = pos.z * (1.0f / depth_fy) * sprite_scale;

    // Build tangent frame from normal
    float3 N = normalize(normal);

    // Choose reference vector not parallel to N
    float3 ref_vec = (abs(N.y) < 0.9f) ? float3(0.f, 1.f, 0.f) : float3(1.f, 0.f, 0.f);
    float3 T = normalize(cross(ref_vec, N));
    float3 B = cross(N, T);

    // Compute center UV
    float2 center_uv;
    if (has_color_projection)
        center_uv = project_to_color_uv(pos);
    else
        center_uv = uvCoords[idx];

    // Compute hexagon vertex positions and UVs
    float3 hex_world[6] = {};
    float2 hex_uv[6]    = {};

    for (int i = 0; i < HEXAGON_VERTS; i++)
    {
        hex_world[i] = pos + radius * (hex_offsets[i].x * T + hex_offsets[i].y * B);

        if (has_color_projection)
            hex_uv[i] = project_to_color_uv(hex_world[i]);
        else
            hex_uv[i] = center_uv;  // fallback: same UV for all vertices
    }

    float4 center_clip = world_to_clip(pos);

    // Emit hexagon as triangle fan via triangle strips:
    // 6 triangles: center-v0-v1, center-v1-v2, ..., center-v5-v0
    for (int i = 0; i < HEXAGON_VERTS; i++)
    {
        int next = (i + 1) % HEXAGON_VERTS;

        GeometryToFragment out_v;

        // Center vertex (dist = 0: fully opaque)
        out_v.pos = center_clip;
        out_v.uv = center_uv;
        out_v.dist = 0.f;
        outStream.Append(out_v);

        // Current hex vertex (dist = 1: edge)
        out_v.pos = world_to_clip(hex_world[i]);
        out_v.uv = hex_uv[i];
        out_v.dist = 1.f;
        outStream.Append(out_v);

        // Next hex vertex (dist = 1: edge)
        out_v.pos = world_to_clip(hex_world[next]);
        out_v.uv = hex_uv[next];
        out_v.dist = 1.f;
        outStream.Append(out_v);

        outStream.RestartStrip();
    }
}

// --- Fragment shader ---

[shader("fragment")]
float4 fragment_main(GeometryToFragment input) : SV_Target {
    // Discard invalid UVs
    if (input.uv.x < 0.f || input.uv.x > 1.f || input.uv.y < 0.f || input.uv.y > 1.f)
        discard;

    // Distance-based alpha falloff: opaque at center, transparent at edge.
    // pow(dist, n) with n > 1 keeps the center opaque and falls off at the edge.
    float alpha = saturate(1.f - pow(saturate(input.dist), falloff_exponent));

    if (useStaticColor)
    {
        float4 sc = staticColor;
        sc.a *= alpha;
        return sc;
    }

    float4 c = colorTex.Sample(sampler_colorTex, input.uv);
    c.a = alpha;
    return c;
}
