#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "parameters.h"
#include "cuda_vec_math.cuh"

namespace {
    __global__ void KBoundary(float4* pos_pred, float4* vel, sim::GridBounds grid, float restitution, uint32_t N) {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= N) return;
        float3 p = to_float3(pos_pred[i]);
        float3 v = to_float3(vel[i]);
        const float e = fminf(fmaxf(restitution, 0.0f), 1.0f); // clamp [0,1]

        // AABB 碰撞，吸附到边界内侧
        if (p.x < grid.mins.x) { p.x = grid.mins.x; v.x *= -e; }
        if (p.y < grid.mins.y) { p.y = grid.mins.y; v.y *= -e; }
        if (p.z < grid.mins.z) { p.z = grid.mins.z; v.z *= -e; }
        if (p.x > grid.maxs.x) { p.x = grid.maxs.x; v.x *= -e; }
        if (p.y > grid.maxs.y) { p.y = grid.maxs.y; v.y *= -e; }
        if (p.z > grid.maxs.z) { p.z = grid.maxs.z; v.z *= -e; }
        pos_pred[i] = make_float4(p.x, p.y, p.z, 1.0f);
        vel[i] = make_float4(v.x, v.y, v.z, 0.0f);
    }
    __global__ void KVelocity(float4* vel, const float4* pos, const float4* pos_pred, float inv_dt, uint32_t N) {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= N) return;
        float3 p = to_float3(pos[i]);
        float3 pp = to_float3(pos_pred[i]);
        float3 v = make_float3((pp.x - p.x) * inv_dt, (pp.y - p.y) * inv_dt, (pp.z - p.z) * inv_dt);
        vel[i] = make_float4(v.x, v.y, v.z, 0.0f);
    }
}

extern "C" void LaunchBoundary(float4* pos_pred, float4* vel, sim::GridBounds grid, float restitution, uint32_t N, cudaStream_t s) {
    const int BS = 256; dim3 b(BS), g((N + BS - 1) / BS);
    KBoundary << <g, b, 0, s >> > (pos_pred, vel, grid, restitution, N);
}
extern "C" void LaunchVelocity(float4* vel, const float4* pos, const float4* pos_pred, float inv_dt, uint32_t N, cudaStream_t s) {
    const int BS = 256; dim3 b(BS), g((N + BS - 1) / BS);
    KVelocity << <g, b, 0, s >> > (vel, pos, pos_pred, inv_dt, N);
}