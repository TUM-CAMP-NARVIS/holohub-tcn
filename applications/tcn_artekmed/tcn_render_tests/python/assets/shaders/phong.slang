uniform float4x4 view;
uniform float4x4 proj;
uniform float4x4 model;

uniform float3 cameraWorldPos;
uniform bool renderStaticColor;

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
    float3 worldNormal : TEXCOORD2;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertex_main(VertexInput input) {
    VertexOutput out;

    float4 worldPos = mul(model, float4(input.pos, 1.0));
    out.worldPos = worldPos.xyz;

    // Transform normal to world space (assuming uniform scale)
    // For non-uniform scale, use the inverse-transpose of the model matrix
    out.worldNormal = normalize(mul((float3x3)model, input.norm));

    out.pos = mul(proj, mul(view, worldPos));
    out.uv = input.uv;

    return out;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target {
    // Basic material properties
    float3 lightPos = float3(-2.0, 5.0, 5.0);
    // Camera position is the inverse of the view matrix translation component
    // In many setups, if we are in world space, we can pass this as a uniform
    // For simplicity, let's assume a static light for now.
    float3 viewPos = float3(5.0, 5.0, 5.0);

    // 1. Texture/Albedo Lookup
    float4 albedo = float4(1.0);
    if (renderStaticColor) {
        albedo = float4(0.8, 0.2, 0.2, 1.0);
    } else {
        albedo = texture.Sample(sampler, input.uv);
    }

    // 2. Ambient
    float ambientStrength = 0.2;
    float3 ambient = ambientStrength * albedo.rgb;

    // 3. Diffuse
    float3 norm = normalize(input.worldNormal);
    float3 lightDir = normalize(lightPos - input.worldPos);
    float diff = max(dot(norm, lightDir), 0.0);
    float3 diffuse = diff * albedo.rgb;

    // 4. Specular (Blinn-Phong)
    float3 viewDir = normalize(viewPos - input.worldPos);
    float3 halfwayDir = normalize(lightDir + viewDir);

    float shininess = 32.0;
    float specularStrength = 0.5;
    float spec = pow(max(dot(norm, halfwayDir), 0.0), shininess);
    float3 specular = specularStrength * spec * float3(1.0, 1.0, 1.0);

    // Final composition
    float3 result = ambient + diffuse + specular;
    return float4(result, albedo.a);
}