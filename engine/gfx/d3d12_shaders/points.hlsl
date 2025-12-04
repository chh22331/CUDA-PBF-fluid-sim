StructuredBuffer<float4> gPositions : register(t0);
StructuredBuffer<float3> gPalette   : register(t1);

cbuffer FrameCB : register(b0)
{
    float4x4 gViewProj;
    float2   gScreenSize;
    float    gParticleRadiusPx;
    float    gThicknessScale;
    uint     gGroups;
    uint     gParticlesPerGroup;
    uint     pad0;
    uint     pad1;
};

struct VSOut {
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD0;
    float3 col : COLOR0;
};

VSOut VSMain(uint vid : SV_VertexID, uint iid : SV_InstanceID)
{
    VSOut o;
    float4 P = gPositions[iid];

    bool hasPalette = (gGroups > 0);
    uint groupIndex = 0;
    if (hasPalette) {
        if (gParticlesPerGroup > 0) {
            groupIndex = iid / gParticlesPerGroup;
        }
        uint wTag = (uint)P.w;
        if (gParticlesPerGroup == 0 && wTag < gGroups) {
            groupIndex = wTag;
        }
        if (groupIndex >= gGroups) groupIndex = 0;
    }
    float3 baseColor = hasPalette && (groupIndex < gGroups)
        ? gPalette[groupIndex]
        : float3(0.55, 0.65, 0.9);

    static const float2 quad[6] = {
        float2(-1,-1), float2(-1, 1), float2(1, 1),
        float2(-1,-1), float2(1, 1), float2(1,-1)
    };

    uint localVid = vid % 6;
    float2 q = quad[localVid];

    float4 clip = mul(gViewProj, float4(P.xyz, 1.0));

    float2 invScreen = 1.0 / gScreenSize;
    float scalePxToClip = clip.w * 2.0f;
    clip.xy += q * gParticleRadiusPx * invScreen * scalePxToClip;

    o.pos = clip;
    o.uv = q * 0.5f + 0.5f;
    o.col = baseColor;
    return o;
}

float4 PSMain(VSOut i) : SV_Target
{
    return float4(i.col, 1.0);
}
