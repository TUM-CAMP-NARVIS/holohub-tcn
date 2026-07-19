import "slangpy";

// Uniforms
uniform float4x4 transform;
uniform float4x4 view;
uniform float4x4 proj;
uniform float4x4 model;

// Point cloud data buffers (separate)
uniform StructuredBuffer<float3> vertices;
uniform StructuredBuffer<float2> uvCoords;
uniform Texture2D<float4> colorTex;
uniform Texture2D<uint> bodyIndexTex;

uniform float4 staticColor;
uniform float pointSize;

uniform float enableClipping;
uniform float clipScale;
uniform float4x4 worldToBox;
uniform float4x4 roI2Mag;
uniform float4 scaleCenter;

uniform float4 boxPosition_1;
uniform float4 boxSize_1;
uniform float4 boxPosition_2;
uniform float4 boxSize_2;
uniform float4 boxPosition_3;
uniform float4 boxSize_3;

uniform int depthWidth;
uniform int depthHeight;
uniform int depthMultiple;
uniform uint processingRadius;

uniform bool useStaticColor;
uniform bool drawUnconnected;
uniform bool hasBodyIndex;
uniform bool dontRenderPlayers;
uniform bool dontRenderEnvironment;
uniform bool activateAdditionalBox;
uniform bool useClippingBox;

uniform float2 screenParams;

uniform SamplerState sampler_colorTex;

// Vertex stage output / Geometry stage input
struct VertexToGeometry {
    uint vid : VERTEXID;
};

// Geometry stage output / Fragment stage input
struct GeometryToFragment {
    float2 uv : TEXCOORD0;
    float4 pos : SV_Position;
    uint vid : VERTEXID;
    //float depth;
};

[shader("vertex")]
VertexToGeometry vertex_main(uint vid : SV_VertexID) {
    VertexToGeometry output;
    output.vid = vid;
    return output;
}

float4 objectToClipPos(float4 objectPos) {
    float4 worldPos = mul(model, objectPos);
    float4 viewPos = mul(view, worldPos);
    return mul(proj, viewPos);
}

// convert to view-direction z+
inline float3 get_pos(uint idx) {
    return vertices[idx];
}

[shader("geometry")]
[maxvertexcount(6)]
void geometry_main(
    point VertexToGeometry input[1],
    inout TriangleStream<GeometryToFragment> outStream)
{
    uint index = input[0].vid;
    uint2 texel = uint2(index % depthWidth, index / depthWidth);

    if (hasBodyIndex) {
        uint bodyIndex = bodyIndexTex.Load(int3(texel, 0));

        if (dontRenderPlayers && bodyIndex != 255)
            return;

        if (dontRenderEnvironment && bodyIndex == 255)
            return;
    }

    GeometryToFragment output;
    output.vid = input[0].vid;
    //output.depth = 0.f;

    if (drawUnconnected) {
        // Unconnected point cloud rendering (billboard sprites)
        uint idx = input[0].vid;
        float3 pos = get_pos(idx);
        //output.depth = sqrt(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z);
        float2 uv = uvCoords[idx];

        float4 origin = float4(pos, 1.0);
        origin = mul(transform, origin);

        if (enableClipping > 0.5f) {
            float4 scaleOffset = scaleCenter - clipScale * scaleCenter;
            origin = clipScale * origin + scaleOffset;
        }

        origin = objectToClipPos(origin);

        float dx = pointSize;
        float dy = pointSize * screenParams.x / screenParams.y;

        output.uv = uv;

        // First triangle
        output.pos = origin + float4(-dx, -dy, 0, 0);
        outStream.Append(output);
        output.pos = origin + float4(dx, -dy, 0, 0);
        outStream.Append(output);
        output.pos = origin + float4(dx, dy, 0, 0);
        outStream.Append(output);

        outStream.RestartStrip();

        // Second triangle
        output.pos = origin + float4(-dx, dy, 0, 0);
        outStream.Append(output);
        output.pos = origin + float4(-dx, -dy, 0, 0);
        outStream.Append(output);
        output.pos = origin + float4(dx, dy, 0, 0);
        outStream.Append(output);

        outStream.RestartStrip();
    }
    else {
        // Surfaced shader rendering (mesh triangulation)
        if ((texel[0] + 1) == (uint)depthWidth || texel[1] == (uint)depthHeight - 1 ||
            (texel[0] - 1) < 0 || (texel[1] - 1) < 0)
            return;

        // Load 3x3 neighborhood positions
        float3 v00_pos = get_pos(index - depthWidth - 1);
        float3 v01_pos = get_pos(index - depthWidth);
        float3 v02_pos = get_pos(index - depthWidth + 1);

        float3 v10_pos = get_pos(index - 1);
        float3 v11_pos = get_pos(index);
        float3 v12_pos = get_pos(index + 1);

        float3 v20_pos = get_pos(index + depthWidth - 1);
        float3 v21_pos = get_pos(index + depthWidth);
        float3 v22_pos = get_pos(index + depthWidth + 1);

        // Load 3x3 neighborhood UV coordinates
        float2 v00_uv = uvCoords[index - depthWidth - 1];
        float2 v01_uv = uvCoords[index - depthWidth];
        float2 v02_uv = uvCoords[index - depthWidth + 1];

        float2 v10_uv = uvCoords[index - 1];
        float2 v11_uv = uvCoords[index];
        float2 v12_uv = uvCoords[index + 1];

        float2 v20_uv = uvCoords[index + depthWidth - 1];
        float2 v21_uv = uvCoords[index + depthWidth];
        float2 v22_uv = uvCoords[index + depthWidth + 1];

        float cullDistance = 10;

        // Depth culling
        if (v00_pos.z < 0.1 || v01_pos.z < 0.1 || v02_pos.z < 0.1 ||
            v10_pos.z < 0.1 || v11_pos.z < 0.1 || v12_pos.z < 0.1 ||
            v20_pos.z < 0.1 || v21_pos.z < 0.1 || v22_pos.z < 0.1 ||
            v00_pos.z > cullDistance || v01_pos.z > cullDistance || v02_pos.z > cullDistance ||
            v10_pos.z > cullDistance || v11_pos.z > cullDistance || v12_pos.z > cullDistance ||
            v20_pos.z > cullDistance || v21_pos.z > cullDistance || v22_pos.z > cullDistance)
            return;

        // Transform all positions
        float4 uv_00 = mul(transform, float4(v00_pos, 1.0));
        float4 uv_01 = mul(transform, float4(v01_pos, 1.0));
        float4 uv_02 = mul(transform, float4(v02_pos, 1.0));

        float4 uv_10 = mul(transform, float4(v10_pos, 1.0));
        float4 uv_11 = mul(transform, float4(v11_pos, 1.0));
        float4 uv_12 = mul(transform, float4(v12_pos, 1.0));

        float4 uv_20 = mul(transform, float4(v20_pos, 1.0));
        float4 uv_21 = mul(transform, float4(v21_pos, 1.0));
        float4 uv_22 = mul(transform, float4(v22_pos, 1.0));

        if (enableClipping > 0.5f) {
            uv_00 = mul(roI2Mag, uv_00);
            uv_01 = mul(roI2Mag, uv_01);
            uv_02 = mul(roI2Mag, uv_02);

            uv_10 = mul(roI2Mag, uv_10);
            uv_11 = mul(roI2Mag, uv_11);
            uv_12 = mul(roI2Mag, uv_12);

            uv_20 = mul(roI2Mag, uv_20);
            uv_21 = mul(roI2Mag, uv_21);
            uv_22 = mul(roI2Mag, uv_22);

            float4 scaleOffset = scaleCenter - clipScale * scaleCenter;

            uv_00 = clipScale * uv_00 + scaleOffset;
            uv_01 = clipScale * uv_01 + scaleOffset;
            uv_02 = clipScale * uv_02 + scaleOffset;

            uv_10 = clipScale * uv_10 + scaleOffset;
            uv_11 = clipScale * uv_11 + scaleOffset;
            uv_12 = clipScale * uv_12 + scaleOffset;

            uv_20 = clipScale * uv_20 + scaleOffset;
            uv_21 = clipScale * uv_21 + scaleOffset;
            uv_22 = clipScale * uv_22 + scaleOffset;
        }

        uv_00 = objectToClipPos(uv_00);
        uv_01 = objectToClipPos(uv_01);
        uv_02 = objectToClipPos(uv_02);

        uv_10 = objectToClipPos(uv_10);
        uv_11 = objectToClipPos(uv_11);
        uv_12 = objectToClipPos(uv_12);

        uv_20 = objectToClipPos(uv_20);
        uv_21 = objectToClipPos(uv_21);
        uv_22 = objectToClipPos(uv_22);

        float maxDist = 0.0075;

        float4 boxpos1 = boxPosition_1;
        float4 boxsize1 = boxSize_1;
        float4 boxpos2 = boxPosition_2;
        float4 boxsize2 = boxSize_2;
        float4 boxpos3 = boxPosition_3;
        float4 boxsize3 = boxSize_3;

        // Check if neighborhood depth is consistent
        if (abs(v11_pos.z - v00_pos.z) < maxDist && abs(v11_pos.z - v01_pos.z) < maxDist && abs(v11_pos.z - v02_pos.z) < maxDist &&
            abs(v11_pos.z - v10_pos.z) < maxDist && abs(v11_pos.z - v12_pos.z) < maxDist &&
            abs(v11_pos.z - v20_pos.z) < maxDist && abs(v11_pos.z - v21_pos.z) < maxDist && abs(v11_pos.z - v22_pos.z) < maxDist)
        {
            float4 vertpos = mul(transform, float4(v11_pos, 1.0));

            if (activateAdditionalBox) {
                if (vertpos.x > boxpos3.x && vertpos.x < boxpos3.x + boxsize3.x &&
                    vertpos.y > boxpos3.y && vertpos.y < boxpos3.y + boxsize3.y &&
                    vertpos.z > boxpos3.z && vertpos.z < boxpos3.z + boxsize3.z)
                    return;
            }

            if (useClippingBox) {
                if (vertpos.x > boxpos1.x && vertpos.x < boxpos1.x + boxsize1.x &&
                    vertpos.y > boxpos1.y && vertpos.y < boxpos1.y + boxsize1.y &&
                    vertpos.z > boxpos1.z && vertpos.z < boxpos1.z + boxsize1.z)
                    return;

                if (vertpos.x > boxpos2.x && vertpos.x < boxpos2.x + boxsize2.x &&
                    vertpos.y > boxpos2.y && vertpos.y < boxpos2.y + boxsize2.y &&
                    vertpos.z > boxpos2.z && vertpos.z < boxpos2.z + boxsize2.z)
                    return;
            }

            // Emit first triangle
            output.pos = uv_00;
            output.uv = v00_uv;
            outStream.Append(output);

            output.pos = uv_02;
            output.uv = v02_uv;
            outStream.Append(output);

            output.pos = uv_22;
            output.uv = v22_uv;
            outStream.Append(output);

            outStream.RestartStrip();
        }

        // Second triangle (same depth consistency check)
        if (abs(v11_pos.z - v00_pos.z) < maxDist && abs(v11_pos.z - v01_pos.z) < maxDist && abs(v11_pos.z - v02_pos.z) < maxDist &&
            abs(v11_pos.z - v10_pos.z) < maxDist && abs(v11_pos.z - v12_pos.z) < maxDist &&
            abs(v11_pos.z - v20_pos.z) < maxDist && abs(v11_pos.z - v21_pos.z) < maxDist && abs(v11_pos.z - v22_pos.z) < maxDist)
        {
            float4 vertpos = mul(transform, float4(v11_pos, 1.0));

            if (activateAdditionalBox) {
                if (vertpos.x > boxpos3.x && vertpos.x < boxpos3.x + boxsize3.x &&
                    vertpos.y > boxpos3.y && vertpos.y < boxpos3.y + boxsize3.y &&
                    vertpos.z > boxpos3.z && vertpos.z < boxpos3.z + boxsize3.z)
                    return;
            }

            if (useClippingBox) {
                if (vertpos.x > boxpos1.x && vertpos.x < boxpos1.x + boxsize1.x &&
                    vertpos.y > boxpos1.y && vertpos.y < boxpos1.y + boxsize1.y &&
                    vertpos.z > boxpos1.z && vertpos.z < boxpos1.z + boxsize1.z)
                    return;

                if (vertpos.x > boxpos2.x && vertpos.x < boxpos2.x + boxsize2.x &&
                    vertpos.y > boxpos2.y && vertpos.y < boxpos2.y + boxsize2.y &&
                    vertpos.z > boxpos2.z && vertpos.z < boxpos2.z + boxsize2.z)
                    return;
            }

            // Emit second triangle
            output.pos = uv_00;
            output.uv = v00_uv;
            outStream.Append(output);

            output.pos = uv_22;
            output.uv = v22_uv;
            outStream.Append(output);

            output.pos = uv_20;
            output.uv = v20_uv;
            outStream.Append(output);

            outStream.RestartStrip();
        }
    }
}

[shader("fragment")]
float4 fragment_main(GeometryToFragment input) : SV_Target {

    // discard the points that have no texture..
    if (input.uv.x <= 0. || input.uv.x > 1. || input.uv.y <= 0. || input.uv.y > 1.0) {
        //return float4(0., 0., 1., 0.);
        discard;
    }

    if (enableClipping > 0.5f) {
        float3 pos = vertices[input.vid];
        float4 origin = mul(transform, float4(pos, 1.0));

        float3 boxPosition = mul(worldToBox, origin).xyz;

        // Clip function in Slang
        if (any(boxPosition < -0.5) || any(boxPosition > 0.5))
            discard;
    }

    if (useStaticColor) {
        return staticColor;
    } else {
        float4 c = colorTex.Sample(sampler_colorTex, input.uv);
        c.a = 1.0f;
        //float4 c = float4(input.depth/30.f, 0.f, 0.f, 1.f);
        return c;
    }
}
