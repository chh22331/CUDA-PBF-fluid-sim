#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "parameters.h"
#include "device_buffers.cuh"
#include "precision_traits.cuh"

using namespace sim;

// 半精算术边界：仅对位置修正 + 速度反弹 (rest>0)
// forceFp32 -> 所有计算转 float
__global__ void KBoundaryHalf(
    float4* __restrict__ pos_pred_fp32,
    float4* __restrict__ vel_fp32,
    Half4* __restrict__ pos_pred_h4,
    Half4* __restrict__ vel_h4,
    sim::GridBounds grid,
    float restitution,
    uint32_t N,
    int forceFp32Accum)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float4 p4 = PrecisionTraits::loadPosPred(pos_pred_fp32, pos_pred_h4, i);
    float4 v4 = PrecisionTraits::loadVel(vel_fp32, vel_h4, i);

    float3 mins = grid.mins;
    float3 maxs = grid.maxs;

    auto clampAxis = [&](float& x, float& vx, float lo, float hi) {
        if (x < lo) {
            x = lo;
            if (restitution > 0.f) vx = -vx * restitution;
            else vx = 0.f;
        }
        else if (x > hi) {
            x = hi;
            if (restitution > 0.f) vx = -vx * restitution;
            else vx = 0.f;
        }
    };

    clampAxis(p4.x, v4.x, mins.x, maxs.x);
    clampAxis(p4.y, v4.y, mins.y, maxs.y);
    clampAxis(p4.z, v4.z, mins.z, maxs.z);

    // store back
    PrecisionTraits::storePosPred(pos_pred_fp32, pos_pred_h4, i, make_float3(p4.x,p4.y,p4.z), p4.w);
    PrecisionTraits::storeVel(vel_fp32, vel_h4, i, make_float3(v4.x,v4.y,v4.z));
}

static inline uint32_t gridFor(uint32_t N) { return (N + 255u) / 256u; }

extern "C" void LaunchBoundaryHalf(
    float4* pos_pred,
    float4* vel,
    const sim::Half4* pos_pred_h4_const,
    const sim::Half4* vel_h4_const,
    sim::GridBounds grid,
    float restitution,
    uint32_t N,
    bool forceFp32Accum,
    cudaStream_t s)
{
    if (N == 0) return;
    KBoundaryHalf<<<gridFor(N), 256, 0, s>>>(
        pos_pred, vel, (Half4*)pos_pred_h4_const, (Half4*)vel_h4_const,
        grid, restitution, N, forceFp32Accum ? 1 : 0);
}