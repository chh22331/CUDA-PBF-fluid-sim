#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "parameters.h"
#include "cuda_vec_math.cuh"

namespace {
    __global__ void KIntegratePred(float4* pos, const float4* vel, float4* pos_pred, float3 gravity, float dt, uint32_t N) {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;
        float3 p = to_float3(pos[i]);
        float3 v = to_float3(vel[i]);
        v += gravity * dt;
        float3 pp = p + v * dt;
        pos_pred[i] = make_float4(pp.x, pp.y, pp.z, 1.0f);
    }
    __global__ void KIntegratePred(float4* pos, const float4* vel, float4* pos_pred,
        float3 gravity, float dt, uint32_t N) {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;
        float4 p4 = pos[i];
        float4 v4 = vel[i];
        float3 p = make_float3(p4.x, p4.y, p4.z);
        float3 v = make_float3(v4.x, v4.y, v4.z);
        v += gravity * dt;
        float3 pp = p + v * dt;
        pos_pred[i] = make_float4(pp.x, pp.y, pp.z, p4.w);
    }
}

extern "C" void LaunchIntegratePred(float4* pos, const float4* vel, float4* pos_pred, float3 gravity, float dt, uint32_t N, cudaStream_t s) {
    const int BS = 256;
    dim3 block(BS), grid((N + BS - 1) / BS);
    KIntegratePred<<<grid, block, 0, s>>>(pos, vel, pos_pred, gravity, dt, N);
}