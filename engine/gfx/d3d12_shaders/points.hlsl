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

    // 新逻辑：优先使用位置缓冲的 w 作为组号（播种时写入），保证排序后仍正确
    bool hasPalette = (gGroups > 0);
    uint groupIndex = 0;
    if (hasPalette) {
    // 首选：实例 ID 直接按每团粒子数分组（排序/重排不会失效）
            if (gParticlesPerGroup > 0) {
            groupIndex = iid / gParticlesPerGroup;
        }
        // 兼容：只有在无法通过粒子数推断时才使用 pos.w（播种写入；若内核保持则可用）
        uint wTag = (uint)P.w;
        if (gParticlesPerGroup == 0 && wTag < gGroups) {
            groupIndex = wTag;
        }
        // 额外容错：若 wTag 不合理且通过除法得到的组号越界，钳制
        if (groupIndex >= gGroups) groupIndex = 0;
    }
    float3 baseColor = hasPalette && (groupIndex < gGroups) ? gPalette[groupIndex] : float3(0.55, 0.65, 0.9); // 调整 Faucet 回退色

    // 6 顶点屏幕空间四边形模板（两个三角形）
    static const float2 quad[6] = {
        float2(-1,-1), float2(-1, 1), float2(1, 1),
        float2(-1,-1), float2(1, 1), float2(1,-1)
    };
    // 修复: 顶点 ID 在实例之间不重置, 使用取模获得局部索引
    uint localVid = vid % 6;
    float2 q = quad[localVid];

    // 世界 -> 裁剪
    float4 clip = mul(gViewProj, float4(P.xyz, 1.0));

    // 将像素半径转换为裁剪空间偏移 (屏幕像素 → NDC → clip)
    float2 deltaClip;
    float2 invScreen = float2(1.0 / gScreenSize.x, 1.0 / gScreenSize.y);
    float scalePxToClip = clip.w * 2.0f; // NDC 像素尺寸 * w
    deltaClip.x = q.x * gParticleRadiusPx * invScreen.x * scalePxToClip;
    deltaClip.y = q.y * gParticleRadiusPx * invScreen.y * scalePxToClip;

    clip.x += deltaClip.x;
    clip.y += deltaClip.y;

    o.pos = clip;
    o.uv = q * 0.5f + 0.5f;
    o.col = baseColor;
    return o;
}

float4 PSMain(VSOut i) : SV_Target
{
    // 若需要圆形点，可在此裁剪: if(length(i.uv*2-1)>1) discard;
    return float4(i.col, 1.0);
}

