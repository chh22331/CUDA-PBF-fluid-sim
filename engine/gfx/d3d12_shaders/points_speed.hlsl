StructuredBuffer<float4> gPositions : register(t0);
StructuredBuffer<float4> gVelocities : register(t1);
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
    float4   gAudioDebug0;
    float4   gAudioDebug1;
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

    float2 debugInfo = ComputeAudioDebug(P.xyz);

    o.pos = clip;
    o.uv = q * 0.5f + 0.5f;
    o.col = speedColor;
    o.debugMask = debugInfo.x;
    o.debugKey = debugInfo.y;
    return o;
}

float4 PSMain(VSOut i) : SV_Target
{
    float3 color = ApplyAudioDebug(i.col, i.debugMask, i.debugKey);
    return float4(color, 1.0);
}
