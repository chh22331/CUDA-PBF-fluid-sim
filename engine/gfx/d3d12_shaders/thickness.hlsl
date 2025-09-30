cbuffer PerFrameCB : register(b0)
{
    float4x4 ViewProj;        // world->clip
    float2   ScreenSize;      // (width, height)
    float    ParticleRadiusPx;
    float    ThicknessScale;
};

// 每个实例对应一个粒子，t0: xyz=world pos, w=unused
StructuredBuffer<float4> ParticlePos : register(t0);

struct VSOut
{
    float4 pos : SV_Position;
    float2 local : TEXCOORD0; // -1..1, 用于圆形遮罩/局部参数
};

VSOut VSMain(uint vid : SV_VertexID, uint iid : SV_InstanceID)
{
    // 两个三角形构成的单位正方形(-1..1)
    const float2 quad[6] = {
        float2(-1.0, -1.0), float2(1.0, -1.0), float2(1.0,  1.0),
        float2(-1.0, -1.0), float2(1.0,  1.0), float2(-1.0,  1.0)
    };
    float2 q = quad[vid];

    // 粒子世界坐标
    uint count, stride; ParticlePos.GetDimensions(count, stride);
    float3 Pw = (iid < count) ? ParticlePos[iid].xyz : float3(0, 0, 0);

    // 世界->裁剪空间中心
    float4 Pc = mul(ViewProj, float4(Pw, 1.0));
    float invW = rcp(max(Pc.w, 1e-6));
    float2 ndc = Pc.xy * invW;

    // 像素半径 -> NDC 偏移（注意Y轴为负以适配D3D屏幕空间）
    float2 pxToNdc = float2(2.0 / max(ScreenSize.x, 1.0),
        -2.0 / max(ScreenSize.y, 1.0));
    float2 offsetNdc = q * ParticleRadiusPx * pxToNdc;

    VSOut o;
    o.local = q; // -1..1
    // 直接输出NDC并将w设为1（保持中心深度为Pc.z/Pc.w）
    o.pos = float4(ndc + offsetNdc, Pc.z * invW, 1.0);
    return o;
}

float4 PSMain(VSOut i) : SV_Target
{
    float r2 = dot(i.local, i.local);
    if (r2 > 1.0) discard;

    // 球冠厚度近似（单位半径厚度=1）
    float thickness = ThicknessScale * sqrt(saturate(1.0 - r2));
    return float4(thickness, 0.0, 0.0, 0.0); // R16F
}

// 兼容工程将入口设置为 "main" 的情况（像素着色器）
float4 main(VSOut i) : SV_Target
{
    return PSMain(i);
}