#include <cuda_runtime.h>
#include "parameters.h"

extern "C" __global__
void KIntegratePredSemiImplicit(const float4* __restrict__ pos,
    float4* __restrict__ vel,          // 修改：去掉 const，写回更新后的速度
    float4* __restrict__ pos_pred,
    float3 gravity,
    float dt,
    uint32_t N) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float4 p = pos[i];
    float4 v4 = vel[i];

    // 半隐式：先更新速度再用更新后速度积分位置
    float3 vNew = make_float3(v4.x + gravity.x * dt,
        v4.y + gravity.y * dt,
        v4.z + gravity.z * dt);

    p.x += vNew.x * dt;
    p.y += vNew.y * dt;
    p.z += vNew.z * dt;

    pos_pred[i] = p;
    // 回写速度（保留 w 分量）
    vel[i] = make_float4(vNew.x, vNew.y, vNew.z, v4.w);
}

extern "C" void LaunchIntegratePredSemiImplicit(const float4* pos,
    float4* vel,                        // 修改：去掉 const
    float4* pos_pred,
    float3 gravity,
    float dt,
    uint32_t N,
    cudaStream_t s) {
    if (!pos || !vel || !pos_pred || N == 0) return;
    uint32_t blocks = (N + 255u) / 256u;
    KIntegratePredSemiImplicit << <blocks, 256, 0, s >> > (
        pos, vel, pos_pred, gravity, dt, N);
}