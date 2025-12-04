StructuredBuffer<float4> gPositions : register(t0);
StructuredBuffer<float4> gVelocities : register(t1);

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

float3 heatmapColor(float t)
{
    // perceptual heatmap ramp: deep blue -> cyan -> green -> yellow -> red
    static const float3 ramp[5] = {
        float3(0.043, 0.016, 0.255),
        float3(0.229, 0.559, 0.996),
        float3(0.321, 0.764, 0.435),
        float3(0.969, 0.902, 0.145),
        float3(0.925, 0.215, 0.129)
    };

    float scaled = saturate(t) * 4.0;
    uint idx = (uint)scaled;
    uint nextIdx = min(idx + 1, 4);
    float fracPart = saturate(scaled - idx);
    return lerp(ramp[idx], ramp[nextIdx], fracPart);
}

struct VSOut {
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD0;
    float3 col : COLOR0;
};

VSOut VSMain(uint vid : SV_VertexID, uint iid : SV_InstanceID)
{
    VSOut o;
    float4 P = gPositions[iid];
    float4 V4 = gVelocities[iid];
    float3 V = V4.xyz;

    float speed = length(V);
    float t = saturate(speed * gThicknessScale);
    float3 speedColor = heatmapColor(t);

    static const float2 quad[6] = {
        float2(-1,-1), float2(-1, 1), float2(1, 1),
        float2(-1,-1), float2(1, 1), float2(1,-1)
    };
    uint localVid = vid % 6;
    float2 q = quad[localVid];

    float4 clip = mul(gViewProj, float4(P.xyz, 1.0));

    float2 invScreen = float2(1.0 / gScreenSize.x, 1.0 / gScreenSize.y);
    float scalePxToClip = clip.w * 2.0f;
    float2 deltaClip;
    deltaClip.x = q.x * gParticleRadiusPx * invScreen.x * scalePxToClip;
    deltaClip.y = q.y * gParticleRadiusPx * invScreen.y * scalePxToClip;

    clip.x += deltaClip.x;
    clip.y += deltaClip.y;

    o.pos = clip;
    o.uv = q * 0.5f + 0.5f;
    o.col = speedColor;
    return o;
}

float4 PSMain(VSOut i) : SV_Target
{
    return float4(i.col, 1.0);
}
