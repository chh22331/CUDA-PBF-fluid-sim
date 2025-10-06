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
    float2 quad[6] = {
        float2(-1,-1), float2(1,-1), float2(1,1),
        float2(-1,-1), float2(1,1),  float2(-1,1)
    };

    float4 wp = ParticlePos[iid];

    // 修复：列主序矩阵应使用 M * v（左乘）
    float4 centerClip = mul(ViewProj, float4(wp.xyz, 1.0));

    float2 ndcCenter = centerClip.xy / max(centerClip.w, 1e-6);

    float2 ndcPerPixel = 2.0 / max(ScreenSize, 1.0);
    float2 ndcRadius = ndcPerPixel * ParticleRadiusPx;

    float2 offset = quad[vid] * ndcRadius;

    VSOut o;
    // 直接写入 NDC（w=1），z 也写 NDC 范围
    o.pos = float4(ndcCenter + offset, centerClip.z / max(centerClip.w, 1e-6), 1.0);
    o.uv = quad[vid];
    return o;
}

float4 PSMain(VSOut IN) : SV_Target
{
    float d = length(IN.uv);
    if (d > 1.0) discard; // circle mask

    float alpha = saturate(1.0 - d);
    float3 col = float3(0.1, 0.7, 1.0);
    return float4(col, alpha);
}

// Dummy entry for FXC custom rule
float4 main(float4 pos : POSITION) : POSITION
{
    return float4(0,0,0,1);
}