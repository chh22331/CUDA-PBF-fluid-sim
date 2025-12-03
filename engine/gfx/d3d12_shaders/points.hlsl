StructuredBuffer<float4> gPositions : register(t0);
StructuredBuffer<float3> gPalette   : register(t1);
StructuredBuffer<float>  gAudioKeyIntensities : register(t2);

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
    float4   gAudioDebug0; // (enabled, domainMinX, invDomainWidth, unused)
    float4   gAudioDebug1; // (surfaceY, surfaceFalloff, unused, unused)
};

struct VSOut {
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD0;
    float3 col : COLOR0;
    float  debugMask : TEXCOORD1;
    float  debugKey : TEXCOORD2;
};

float2 ComputeAudioDebug(float3 worldPos)
{
    float enabled = gAudioDebug0.x;
    float invWidth = gAudioDebug0.z;
    float domainWidth = (invWidth > 0.0f) ? rcp(invWidth) : 0.0f;
    float domainMin = gAudioDebug0.y;
    float domainMax = domainMin + domainWidth;
    float insideDomain = (enabled > 0.5f && domainWidth > 0.0f && worldPos.x >= domainMin && worldPos.x <= domainMax) ? 1.0f : 0.0f;

    float falloff = max(gAudioDebug1.y, 1e-4f);
    float surfaceStart = gAudioDebug1.x - falloff;
    float surfaceMask = saturate((worldPos.y - surfaceStart) / falloff);

    float keyCount = gAudioDebug1.z;
    float debugKey = saturate((worldPos.x - domainMin) * invWidth);
    uint keyIndex = 0;
    float keyLevel = 0.0f;
    if (keyCount > 0.5f) {
        float scaled = min(debugKey * keyCount, max(keyCount - 1.0f, 0.0f));
        keyIndex = (uint)scaled;
        keyLevel = saturate(gAudioKeyIntensities[keyIndex]);
    }
    float debugMask = insideDomain * surfaceMask * keyLevel;
    return float2(debugMask, debugKey);
}

float3 ApplyAudioDebug(float3 baseColor, float debugMask, float debugKey)
{
    if (gAudioDebug0.x <= 0.5f) return baseColor;
    float3 overlay = lerp(float3(0.15, 0.4, 1.0), float3(1.0, 0.1, 0.8), saturate(debugKey));
    float intensity = saturate(debugMask);
    return lerp(baseColor, overlay, intensity);
}

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
    float2 debugInfo = ComputeAudioDebug(P.xyz);
    o.debugMask = debugInfo.x;
    o.debugKey = debugInfo.y;
    return o;
}

float4 PSMain(VSOut i) : SV_Target
{
    float3 color = ApplyAudioDebug(i.col, i.debugMask, i.debugKey);
    return float4(color, 1.0);
}

