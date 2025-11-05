#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "device_globals.cuh"
#include "cuda_vec_math.cuh"
#include "precision_traits.cuh"

namespace {
    __global__ void KVelocityGlobals(float dtInv, uint32_t N) {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;

        float4 c4 = sim::g_pos[i];
        float4 n4 = sim::g_pos_pred[i];

        float3 c = make_float3(c4.x, c4.y, c4.z);
        float3 n = make_float3(n4.x, n4.y, n4.z);
        float3 v = make_float3((n.x - c.x) * dtInv,
                               (n.y - c.y) * dtInv,
                               (n.z - c.z) * dtInv);

        // 写回 float4 速度 + 可选 half
        sim::PrecisionTraits::storeVel(sim::g_vel, sim::g_vel_h4, i, v);
    }
} // namespace

extern "C" void LaunchVelocityGlobals(float dtInv,
                                      uint32_t N,
                                      cudaStream_t s)
{
    if (N == 0) return;
    const int BS = 256;
    dim3 block(BS), grid((N + BS - 1) / BS);
    KVelocityGlobals<<<grid, block, 0, s>>>(dtInv, N);
}