uniform float4x4 view;
uniform float4x4 proj;
uniform float4x4 model;

uniform bool renderStaticColor;
uniform float pointSize;

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
    float PSize : SV_PointSize;
    uint vid: TEXCOORD1;
    float vpz : TEXCOORD2;
};

[shader("vertex")]
VertexOutput vertex_main(VertexInput input, uint vid: SV_VertexID) {

    VertexOutput out;
    float4 pos = float4(input.pos, 1.0);
    float4 worldPos = mul(model, pos);
    float4 viewPos = mul(view, worldPos);
    out.pos = mul(proj, viewPos);
    out.PSize = pointSize;
    out.vid = vid;
    out.vpz = viewPos.z / -9.;

    //printf("Transformed position = %v4f\n worldPos = %v4f\n viewPos = %v4f\n", out.pos, worldPos, viewPos);

    //float3 worldNormal = normalize(mul((float3x3)model, in.norm));
    //out.normal = worldNormal;

    out.uv = input.uv;
    return out;
}

uint hash_u32(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

float3 id_to_rgb(uint id)
{
    uint h = hash_u32(id);
    // take 8 bits per channel
    float r = float((h      ) & 255) / 255.0;
    float g = float((h >>  8) & 255) / 255.0;
    float b = float((h >> 16) & 255) / 255.0;
    return float3(r, g, b);
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target {

    //float3 N = normalize(in.normal);

    printf("vpz = %f\n", input.vpz);

    // discard the points that have no texture..
    if (input.uv.x <= 0. || input.uv.x > 1. || input.uv.y <= 0. || input.uv.y > 1.0) {
        //return float4(0., 0., 1., 1.);
        discard;
    }

    if (renderStaticColor) {
//         float z = -input.vpz;
//         float z01 = saturate((z - 1.0) / (10. - 1.));
        //float bins = 10.0;
        //float q = floor(z01 * bins) / (bins - 1.0);
        //z01 = sqrt(z01);
//         return float4(z01.xxx, 1.);
        float3 c = id_to_rgb(input.vid);
        return float4(c, 1.0);

    } else {
        float4 texColor = texture.Sample(sampler, input.uv);
        return float4(texColor.rgb, 1.0);
    }
}
