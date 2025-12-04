cbuffer PerFrameCB : register(b0)
{
    float4x4 ViewProj; 
    float2   ScreenSize;  
    float    ParticleRadiusPx;
    float    ThicknessScale;
};
 
StructuredBuffer<float4> ParticlePos : register(t0);

struct VSOut
{
    float4 pos : SV_Position;
    float2 local : TEXCOORD0;  
};

VSOut VSMain(uint vid : SV_VertexID, uint iid : SV_InstanceID)
{
    const float2 quad[6] = {
        float2(-1.0, -1.0), float2(1.0, -1.0), float2(1.0,  1.0),
        float2(-1.0, -1.0), float2(1.0,  1.0), float2(-1.0,  1.0)
    };
    float2 q = quad[vid];
 
    uint count, stride; ParticlePos.GetDimensions(count, stride);
    float3 Pw = (iid < count) ? ParticlePos[iid].xyz : float3(0, 0, 0);
 
    float4 Pc = mul(ViewProj, float4(Pw, 1.0));
    float invW = rcp(max(Pc.w, 1e-6));
    float2 ndc = Pc.xy * invW;
 
    float2 pxToNdc = float2(2.0 / max(ScreenSize.x, 1.0),
        -2.0 / max(ScreenSize.y, 1.0));
    float2 offsetNdc = q * ParticleRadiusPx * pxToNdc;

    VSOut o;
    o.local = q;
    o.pos = float4(ndc + offsetNdc, Pc.z * invW, 1.0);
    return o;
}

float4 PSMain(VSOut i) : SV_Target
{
    float r2 = dot(i.local, i.local);
    if (r2 > 1.0) discard;
 
    float thickness = ThicknessScale * sqrt(saturate(1.0 - r2));
    return float4(thickness, 0.0, 0.0, 0.0);
}

float4 main(VSOut i) : SV_Target
{
    return PSMain(i);
}