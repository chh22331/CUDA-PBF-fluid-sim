cbuffer PerFrameCB : register(b0)
{
    float4x4 ViewProj;
    float2   ScreenSize;
    float    ParticleRadiusPx;
    float    ThicknessScale; // unused here
};

StructuredBuffer<float4> ParticlePos : register(t0); // xyz: world pos, w: unused

struct VSOut {
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0; // local quad uv in [-1,1]
};

VSOut VSMain(uint vid : SV_VertexID, uint iid : SV_InstanceID)
{
    // Expand a quad (two triangles, 6 verts) per instance from SV_VertexID
    // pattern: ( -1,-1 ), ( 1,-1 ), ( 1, 1 ),  ( -1,-1 ), ( 1, 1 ), ( -1, 1 )
    float2 quad[6] = {
        float2(-1,-1), float2(1,-1), float2(1,1),
        float2(-1,-1), float2(1,1),  float2(-1,1)
    };

    float4 wp = ParticlePos[iid];

    // Project center to clip space
    float4 centerClip = mul(float4(wp.xyz, 1.0), ViewProj);
    float2 ndcCenter = centerClip.xy / max(centerClip.w, 1e-6);

    // Convert pixel radius to NDC radius
    float2 ndcPerPixel = 2.0 / max(ScreenSize, 1.0);
    float2 ndcRadius = ndcPerPixel * ParticleRadiusPx;

    float2 offset = quad[vid] * ndcRadius;

    VSOut o;
    o.pos = float4(ndcCenter + offset, centerClip.z / max(centerClip.w, 1e-6), 1.0);
    o.uv = quad[vid];
    return o;
}

float4 PSMain(VSOut IN) : SV_Target
{
    float d = length(IN.uv);
    if (d > 1.0) discard; // circle mask

    // simple alpha falloff for visibility
    float alpha = saturate(1.0 - d);
    float3 col = float3(0.1, 0.7, 1.0);
    return float4(col, alpha);
}

// Dummy entry to satisfy FXC custom build rules that expect an entry named 'main'.
// Compile it as a minimal vertex shader so it succeeds under default VS compilation.
float4 main(float4 pos : POSITION) : POSITION
{
    return float4(0,0,0,1);
}