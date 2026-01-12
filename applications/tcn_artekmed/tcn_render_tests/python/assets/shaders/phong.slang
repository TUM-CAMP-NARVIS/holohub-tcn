uniform float4x4 view;
uniform float4x4 proj;
uniform float4x4 model;

uniform Texture2D<float4> texture;
uniform SamplerState sampler;

struct VertexInput {
    float3 pos : POSITION;
    float3 norm : NORMAL;
    float2 uv : TEXCOORD;
};

struct VertexOutput {
    float4 pos : SV_Position;
    float3 worldPos : TEXCOORD1;
    float3 normal : TEXCOORD2;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertex_main(VertexInput input) {
    VertexOutput out;
    float4 worldPos = mul(model, float4(input.pos, 1.0));
    float4 viewPos = mul(view, worldPos);
    out.pos = mul(proj, viewPos);
    out.worldPos = worldPos.xyz;

    out.normal = input.norm;

    out.uv = input.uv;
    return out;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target {
    // this is not complete or correct ..
    float3 lightPos = {-2, 5, 5};
    float3 cameraPos = { 0, 0, 0 };

    float3 N = normalize(input.normal);
    float3 L = normalize(lightPos - input.worldPos);
    float3 V = normalize(cameraPos - input.worldPos);
    float3 R = reflect(-L, N);

    float ambientStrength = 0.1;
    float3 ambient = ambientStrength * float3(1.0, 1.0, 1.0);

    float diff = max(dot(N, L), 0.0);
    float3 diffuse = diff * float3(1.0, 1.0, 1.0);

    float specularStrength = 0.5;
    float shininess = 32.0;
    float spec = pow(max(dot(R, V), 0.0), shininess);
    float3 specular = specularStrength * spec * float3(1.0, 1.0, 1.0);

    float3 lighting = ambient + diffuse + specular;

    float4 texColor = texture.Sample(sampler, input.uv);
    return float4(texColor.rgb * lighting, texColor.a);
}
