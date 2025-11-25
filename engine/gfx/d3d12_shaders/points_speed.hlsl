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

    // 颜色映射：按速度长度线性映射（蓝 -> 红）
    float speed = length(V);

    // 使用 gThicknessScale 作为速度放大系数的简易输入（外部可把 1/maxSpeed 写入这里）
    float t = saturate(speed * gThicknessScale);

    float3 colorBlue = float3(0.0, 0.6, 1.0);
    float3 colorRed = float3(1.0, 0.0, 0.0);
    float3 speedColor = lerp(colorBlue, colorRed, t);

    // 顶点四边形模板（两个三角形）
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