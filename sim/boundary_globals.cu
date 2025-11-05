#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "device_globals.cuh"
#include "parameters.h"
#include "precision_traits.cuh"

namespace {
__global__ void KBoundaryGlobals(float restitution, sim::GridBounds grid, uint32_t N, bool useXsph) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float4 p4 = sim::PrecisionTraits::loadPosPred(sim::g_pos_pred, sim::g_pos_pred_h4, i);
    float3 v = useXsph
        ? make_float3(sim::g_delta[i].x, sim::g_delta[i].y, sim::g_delta[i].z)
        : make_float3(sim::g_vel[i].x, sim::g_vel[i].y, sim::g_vel[i].z);

    // Clamp + reflect
    if (p4.x < grid.mins.x) { p4.x = grid.mins.x; v.x = -v.x * restitution; }
    else if (p4.x > grid.maxs.x) { p4.x = grid.maxs.x; v.x = -v.x * restitution; }
    if (p4.y < grid.mins.y) { p4.y = grid.mins.y; v.y = -v.y * restitution; }
    else if (p4.y > grid.maxs.y) { p4.y = grid.maxs.y; v.y = -v.y * restitution; }
    if (p4.z < grid.mins.z) { p4.z = grid.mins.z; v.z = -v.z * restitution; }
    else if (p4.z > grid.maxs.z) { p4.z = grid.maxs.z; v.z = -v.z * restitution; }

    sim::PrecisionTraits::storePosPred(sim::g_pos_pred, sim::g_pos_pred_h4, i,
                                       make_float3(p4.x,p4.y,p4.z), p4.w);

    if (useXsph)
        sim::g_delta[i] = make_float4(v.x,v.y,v.z,0.f);

    // 统一写入最终速度
    sim::PrecisionTraits::storeVel(sim::g_vel, sim::g_vel_h4, i, v);
}
} // namespace

extern "C" void LaunchBoundaryGlobals(sim::GridBounds grid,
                                      float restitution,
                                      uint32_t N,
                                      bool xsphApplied,
                                      cudaStream_t s)
{
    if (N==0) return;
    const int BS=256;
    dim3 block(BS), gridDim((N+BS-1)/BS);
    KBoundaryGlobals<<<gridDim, block, 0, s>>>(restitution, grid, N, xsphApplied);
}