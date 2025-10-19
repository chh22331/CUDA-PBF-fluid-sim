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
}

extern "C" void LaunchIntegratePred(float4* pos, const float4* vel, float4* pos_pred, float3 gravity, float dt, uint32_t N, cudaStream_t s) {
    const int BS = 256;
    dim3 block(BS), grid((N + BS - 1) / BS);
    KIntegratePredGlobals << <grid, block, 0, s >> > (gravity, dt, N);
}