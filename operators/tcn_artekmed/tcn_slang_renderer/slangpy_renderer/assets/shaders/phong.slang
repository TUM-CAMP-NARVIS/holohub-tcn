// -----------------------------------------------------------------------------
// Simple Blinn-Phong (single directional light) for Slang/HLSL-style semantics.
// Uses:
//  - view/proj/model for transforms
//  - inverseView to reconstruct camera position in world space
//  - texture + sampler for albedo
//  - renderStaticColor to bypass lighting/texture (useful for debugging)
// -----------------------------------------------------------------------------

uniform float4x4 view;
uniform float4x4 proj;
uniform float4x4 model;
uniform float4x4 inverseView;

uniform bool renderStaticColor;

uniform Texture2D<float4> texture;
uniform SamplerState sampler;

struct VertexInput {
    float3 pos  : POSITION;
    float3 norm : NORMAL;
    float2 uv   : TEXCOORD;
};

struct VertexOutput {
    float4 pos        : SV_Position;
    float3 worldPos   : TEXCOORD1;
    float3 worldNormal: TEXCOORD2;
    float2 uv         : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertex_main(VertexInput input)
{
    VertexOutput out;

    // Object -> World
    float4 wp = mul(model, float4(input.pos, 1.0));
    out.worldPos = wp.xyz;

    // Normal -> World
    // NOTE: This is correct only for rigid/uniform-scale transforms.
    // For non-uniform scale, use inverse-transpose(model) 3x3.
    out.worldNormal = normalize(mul((float3x3)model, input.norm));

    // World -> Clip
    float4 vp = mul(view, wp);
    out.pos = mul(proj, vp);

    out.uv = input.uv;
    return out;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target
{
    // Debug path: constant color (also avoids NaNs if normals are bad)
    if (renderStaticColor)
    {
        // Encode normal for quick debugging (mapped to 0..1)
        float3 ndbg = 0.5 * (normalize(input.worldNormal) + 1.0);
        return float4(ndbg, 1.0);
    }

    // --- Material ---
    float4 texel = texture.Sample(sampler, input.uv);
    float3 albedo = texel.rgb;
    float  alpha  = texel.a;

    // --- Shading frame ---
    float3 N = normalize(input.worldNormal);

    // Camera position from inverseView (assuming rigid view matrix)
    float3 cameraPos = inverseView[3].xyz;

    float3 V = normalize(cameraPos - input.worldPos);

    // --- Light (directional) ---
    // Light direction *from the surface toward the light*
    // Choose something reasonable; make uniform if you want it configurable.
    float3 L = normalize(float3(0.4, 0.8, 0.2));

    // --- Blinn-Phong terms ---
    float NdotL = saturate(dot(N, L));
    float3 H = normalize(L + V);
    float NdotH = saturate(dot(N, H));

    // Tunables (could be uniforms later)
    float3 ambientColor  = float3(0.03, 0.03, 0.03);
    float3 lightColor    = float3(1.0, 1.0, 1.0);
    float  kd            = 1.0;     // diffuse strength
    float  ks            = 0.25;    // specular strength
    float  shininess     = 64.0;    // specular exponent

    float3 ambient  = ambientColor * albedo;
    float3 diffuse  = kd * NdotL * albedo * lightColor;
    float3 specular = ks * pow(NdotH, shininess) * lightColor;

    float3 color = ambient + diffuse + specular;

    return float4(color, alpha);
}