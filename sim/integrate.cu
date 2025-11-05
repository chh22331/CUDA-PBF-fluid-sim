#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "parameters.h"
#include "cuda_vec_math.cuh"
#include "device_globals.cuh"
#include "precision_traits.cuh"

namespace {

    __global__ void KIntegratePredGlobals(float3 gravity, float dt, uint32_t N) {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;

        float4 p4 = sim::g_pos[i];
        float4 v4 = sim::g_vel[i];
        float3 p = make_float3(p4.x, p4.y, p4.z);
        float3 v = make_float3(v4.x, v4.y, v4.z);
        v.x += gravity.x * dt;
        v.y += gravity.y * dt;
        v.z += gravity.z * dt;

        float3 pp = make_float3(p.x + v.x * dt,
            p.y + v.y * dt,
            p.z + v.z * dt);

        // 写预测位置（自动 half 分支）
        sim::PrecisionTraits::storePosPred(sim::g_pos_pred,
            sim::g_pos_pred_h4,
            i, pp, p4.w);
    }

} // namespace

extern "C" void LaunchIntegratePredGlobals(float3 gravity,
    float dt,
    uint32_t N,
    cudaStream_t s)
{
    if (N == 0) return;
    const int BS = 256;
    dim3 block(BS), grid((N + BS - 1) / BS);
    KIntegratePredGlobals << <grid, block, 0, s >> > (gravity, dt, N);
}