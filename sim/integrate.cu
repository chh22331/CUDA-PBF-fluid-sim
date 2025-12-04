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
        float3 p = make_float3(p4.x, p4.y, p4.z);
        float3 v = make_float3(v4.x, v4.y, v4.z);
        v.x += gravity.x * dt;
        v.y += gravity.y * dt;
        v.z += gravity.z * dt;

        float3 pp = make_float3(p.x + v.x * dt,
            p.y + v.y * dt,
            p.z + v.z * dt);
    }
} // namespace

extern "C" {

    __global__ void KIntegratePred(
        const float4* __restrict__ posCurr,
        const float4* __restrict__ velSrc,
        float4* __restrict__ posPred,
        float3 gravity,
        float dt,
        uint32_t N)
    {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;

        float4 p4 = posCurr[i];
        float4 v4 = velSrc[i];
 
        float vx = v4.x + gravity.x * dt;
        float vy = v4.y + gravity.y * dt;
        float vz = v4.z + gravity.z * dt;

        float3 pp = make_float3(p4.x + vx * dt,
            p4.y + vy * dt,
            p4.z + vz * dt);
 
        posPred[i] = make_float4(pp.x, pp.y, pp.z, p4.w);
    }

    __global__ void KVelocity(
        float4* __restrict__ velOut,
        const float4* __restrict__ posCurr,
        const float4* __restrict__ posPred,
        float invDt,
        uint32_t N)
    {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;

        float4 c = posCurr[i];
        float4 p = posPred[i];

        float vx = (p.x - c.x) * invDt;
        float vy = (p.y - c.y) * invDt;
        float vz = (p.z - c.z) * invDt;

        float4 vOut = velOut[i];
        vOut.x = vx; vOut.y = vy; vOut.z = vz;
        velOut[i] = vOut;
    }

    void LaunchIntegratePred(
        float4* posCurr,
        const float4* velSrc,
        float4* posPred,
        float3 gravity,
        float dt,
        uint32_t N,
        cudaStream_t s)
    {
        if (!posCurr || !velSrc || !posPred || N == 0) return;
        const uint32_t BS = 256;
        dim3 block(BS), grid((N + BS - 1) / BS);
        KIntegratePred << <grid, block, 0, s >> > (posCurr, velSrc, posPred, gravity, dt, N);
    }
}