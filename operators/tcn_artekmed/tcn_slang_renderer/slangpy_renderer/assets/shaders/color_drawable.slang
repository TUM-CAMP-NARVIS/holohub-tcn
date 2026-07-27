/*
 * Shader for rendering reference frame line segments with per-vertex colors.
 *
 * Renders colored lines representing coordinate axes:
 * - X axis: Red
 * - Y axis: Green
 * - Z axis: Blue
 */

// Uniforms for transformation matrices
uniform float4x4 proj;
uniform float4x4 view;
uniform float4x4 model;

// Vertex shader input
struct VertexInput {
    float3 position : POSITION;
    float3 color : COLOR;
};

// Vertex shader output / Fragment shader input
struct VertexOutput {
    float4 position : SV_Position;
    float3 color : COLOR;
};

// Vertex shader
[shader("vertex")]
VertexOutput vertex_main(VertexInput input) {
    VertexOutput output;

    // Transform vertex position
    float4 worldPos = mul(model, float4(input.position, 1.0));
    float4 viewPos = mul(view, worldPos);
    output.position = mul(proj, viewPos);

    // Pass through color
    output.color = input.color;

    return output;
}

// Fragment shader
[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target {
    // Simply output the interpolated color
    return float4(input.color, 1.0);
}
