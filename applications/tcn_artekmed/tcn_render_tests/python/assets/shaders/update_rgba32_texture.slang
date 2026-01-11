// shader utils
StructuredBuffer<float4> sourceBuffer;
uniform int width;
uniform int height;

RWTexture2D<float4> textureHandle;

// Copy a linear buffer of float4 texels into a RW texture.
// Assumes src is in row-major order: src[y * width + x].
[shader("compute")]
[numthreads(16,16,1)]
void upload_rgba32f(uint3 threadId : SV_DispatchThreadID)
{
    uint2 p = threadId.xy;
    if (p.x >= width || p.y >= height)
        return;

    uint idx = p.y * width + p.x;
    textureHandle[p] = sourceBuffer[idx];
}