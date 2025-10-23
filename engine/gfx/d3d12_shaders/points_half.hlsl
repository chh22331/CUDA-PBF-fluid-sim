StructuredBuffer<uint2> gPositionsHalf : register(t0); // 每元素8 字节，打包4x fp16 (x,y,z,w)
StructuredBuffer<float3> gPalette : register(t1);

cbuffer FrameCB : register(b0)
{
 float4x4 gViewProj;
 float2 gScreenSize;
 float gParticleRadiusPx;
 float gThicknessScale;
 uint gGroups;
 uint gParticlesPerGroup;
 uint pad0;
 uint pad1;
};

float4 DecodeHalf4(uint2 h){
 uint lo = h.x; uint hi = h.y;
 uint hx = (lo &0xFFFF); uint hy = (lo >>16) &0xFFFF;
 uint hz = (hi &0xFFFF); uint hw = (hi >>16) &0xFFFF;
 float x = f16tof32(hx); float y = f16tof32(hy); float z = f16tof32(hz); float w = f16tof32(hw);
 return float4(x,y,z,w);
}

struct VSOut { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; float3 col : COLOR0; };

VSOut VSMain(uint vid : SV_VertexID, uint iid : SV_InstanceID)
{
 VSOut o;
 float4 P = DecodeHalf4(gPositionsHalf[iid]);
 bool hasPalette = (gGroups >0);
 uint groupIndex =0;
 if (hasPalette) {
 if (gParticlesPerGroup >0) groupIndex = iid / gParticlesPerGroup;
 uint wTag = (uint)P.w;
 if (gParticlesPerGroup ==0 && wTag < gGroups) groupIndex = wTag;
 if (groupIndex >= gGroups) groupIndex =0;
 }
 float3 baseColor = hasPalette && (groupIndex < gGroups) ? gPalette[groupIndex] : float3(0.55,0.65,0.9);
 static const float2 quad[6] = {
 float2(-1,-1), float2(-1,1), float2(1,1),
 float2(-1,-1), float2(1,1), float2(1,-1)
 };
 uint localVid = vid %6; float2 q = quad[localVid];
 float4 clip = mul(gViewProj, float4(P.xyz,1));
 float2 invScreen =1.0 / gScreenSize;
 float scalePxToClip = clip.w *2.0;
 float2 deltaClip = float2(q.x * gParticleRadiusPx * invScreen.x * scalePxToClip,
 q.y * gParticleRadiusPx * invScreen.y * scalePxToClip);
 clip.xy += deltaClip;
 o.pos = clip; o.uv = q*0.5+0.5; o.col = baseColor; return o;
}

float4 PSMain(VSOut i) : SV_Target { return float4(i.col,1); }

