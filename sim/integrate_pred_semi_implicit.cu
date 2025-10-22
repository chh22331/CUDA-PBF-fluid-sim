#include <cuda_runtime.h>
#include "parameters.h"

extern "C" __global__
void KIntegratePredSemiImplicit(const float4* __restrict__ pos,
    const float4* __restrict__ vel,
    float4* __restrict__ pos_pred,
    float3 gravity,
    float dt,
    uint32_t N) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float4 p = pos[i];
    float4 v = vel[i];
    // 半隐式：先更新速度再用更新后速度积分位置（速度本身不写回）
    float vx_dt = (v.x + gravity.x * dt) * dt;
    float vy_dt = (v.y + gravity.y * dt) * dt;
    float vz_dt = (v.z + gravity.z * dt) * dt;
    p.x += vx_dt; p.y += vy_dt; p.z += vz_dt;
    pos_pred[i] = p;
}

extern "C" void LaunchIntegratePredSemiImplicit(const float4* pos,
    const float4* vel,
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