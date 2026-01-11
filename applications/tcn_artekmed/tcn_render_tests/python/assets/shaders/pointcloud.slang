uniform float4x4 view;
uniform float4x4 proj;
uniform float4x4 model;

uniform bool renderStaticColor;

uniform Texture2D<float4> texture;
uniform SamplerState sampler;

struct VertexInput {
    float3 pos : POSITION;
    float2 uv : TEXCOORD;
};

struct VertexOutput {
    float4 pos : SV_Position;
    //float3 normal : TEXCOORD2;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertex_main(VertexInput input) {

    VertexOutput out;
    float4 worldPos = mul(model, float4(input.pos, 1.0));
    float4 viewPos = mul(view, worldPos);
    out.pos = mul(proj, viewPos);

    //float3 worldNormal = normalize(mul((float3x3)model, in.norm));
    //out.normal = worldNormal;

    out.uv = input.uv;
    return out;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target {

    //float3 N = normalize(in.normal);

    // discard the points that have no texture..
    if (input.uv.x <= 0. || input.uv.x > 1. || input.uv.y <= 0. || input.uv.y > 1.0) {
        //return float4(0., 0., 1., 1.);
        discard;
    }

    if (renderStaticColor) {
        return float4(0.8, 0., 0., 1.);
    } else {
        float4 texColor = texture.Sample(sampler, input.uv);
        return float4(texColor.rgb, texColor.a);
    }
}
