#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "parameters.h"
#include "cuda_vec_math.cuh"
#include "device_globals.cuh"

namespace {
    __global__ void KIntegratePredGlobals(float3 gravity, float dt, uint32_t N) {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;
        float4 p4 = sim::g_pos[i];
        float4 v4 = sim::g_vel[i];
        float3 p = to_float3(p4);
        float3 v = to_float3(v4);
        v += gravity * dt;
        float3 pp = p + v * dt;
        sim::g_pos_pred[i] = make_float4(pp.x, pp.y, pp.z, 1.0f);
    }
   __global__ void KIntegratePred(float4* pos,
                                   const float4* __restrict__ vel,
                                   float4* __restrict__ pos_pred,
                                   float3 gravity,
                                   float dt,
                                   uint32_t N) {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;
        float4 p4 = pos[i];
        float4 v4 = vel[i];
        float3 p = make_float3(p4.x, p4.y, p4.z);
        float3 v = make_float3(v4.x, v4.y, v4.z);
        // v(t+dt) = v(t) + g*dt（不写回），位置增量 = v(t+dt)*dt
        v.x += gravity.x * dt;
        v.y += gravity.y * dt;
        v.z += gravity.z * dt;
        float3 pp = make_float3(p.x + v.x * dt,
                                p.y + v.y * dt,
                                p.z + v.z * dt);
        pos_pred[i] = make_float4(pp.x, pp.y, pp.z, p4.w);
    }
}

extern "C" void LaunchIntegratePred(float4* pos,
                                    const float4* vel,
                                    float4* pos_pred,
                                    float3 gravity,
                                    float dt,
                                    uint32_t N,
                                    cudaStream_t s) {
    if (!pos || !vel || !pos_pred || N == 0) return;
#ifdef _DEBUG
    if (pos_pred == pos) {
        // 运行时保护：预测缓冲不应与当前位置同址，否则后续速度更新 (pos_pred-pos)/dt 恒为 0 或只剩 g*dt。
        printf("[IntegratePred][Warn] pos_pred alias pos (ptr=%p) — check ping-pong swap sequence.\n", pos);
    }
#endif
    const int BS = 256;
    dim3 block(BS), grid((N + BS - 1) / BS);
    KIntegratePred<<<grid, block, 0, s>>>(pos, vel, pos_pred, gravity, dt, N);
}