#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "device_globals.cuh"
#include "cuda_vec_math.cuh"
#include "cuda_grid_utils.cuh"
#include "precision_traits.cuh"
#include "parameters.h"

namespace {

__device__ inline float3 spikyGrad(const float3& rij, float r, const sim::KernelCoeffs& kc) {
    if (r <= 1e-8f) return make_float3(0.f,0.f,0.f);
    float t = kc.h - r;
    if (t <= 0.f) return make_float3(0.f,0.f,0.f);
    float coeff = (-3.f * kc.spiky * t * t / r);
    return make_float3(coeff * rij.x, coeff * rij.y, coeff * rij.z);
}

__global__ void KDeltaApplyGlobalsDense(
    uint32_t N,
    const uint32_t* __restrict__ indicesSorted,
    const uint32_t* __restrict__ keysSorted,
    const uint32_t* __restrict__ cellStart,
    const uint32_t* __restrict__ cellEnd,
    sim::DeviceParams dp)
{
    uint32_t sortedIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sortedIdx >= N) return;

    const auto& grid = dp.grid;
    const auto& kc   = dp.kernel;
    const float restDensity   = dp.restDensity;
    const float mass          = dp.particleMass;
    const auto& pbf           = dp.pbf;

    uint32_t i = indicesSorted[sortedIdx];
    float4 pi4 = sim::PrecisionTraits::loadPosPred(sim::g_pos_pred, sim::g_pos_pred_h4, i);
    float3 pi  = make_float3(pi4.x, pi4.y, pi4.z);
    float lambda_i = sim::g_lambda[i];

    uint32_t key = keysSorted[sortedIdx];
    int3 ci;
    ci.x = int(key % uint32_t(grid.dim.x));
    uint32_t key_div_x = key / uint32_t(grid.dim.x);
    ci.y = int(key_div_x % uint32_t(grid.dim.y));
    ci.z = int(key_div_x / uint32_t(grid.dim.y));

    float3 delta = make_float3(0.f,0.f,0.f);
    int neighborContrib = 0;
    const int cap = dp.maxNeighbors;
    const bool hasCap = (cap > 0);
    const float cs = grid.cellSize;
    const int reach = (cs > 0.f) ? max(1, int(ceilf(kc.h / cs))) : 1;

    for (int dz=-reach; dz<=reach; ++dz)
    for (int dy=-reach; dy<=reach; ++dy)
    for (int dx=-reach; dx<=reach; ++dx) {
        if (hasCap && neighborContrib >= cap) goto Done;
        int3 cc = make_int3(ci.x+dx, ci.y+dy, ci.z+dz);
        if (!sim::inBounds(cc, grid.dim)) continue;
        uint32_t cidx = sim::linIdx(cc, grid.dim);
        uint32_t beg = cellStart[cidx];
        uint32_t end = cellEnd[cidx];
        if (beg == 0xFFFFFFFFu || beg >= end) continue;

        for (uint32_t k=beg; k<end; ++k) {
            if (hasCap && neighborContrib >= cap) break;
            uint32_t j = indicesSorted[k];
            if (j == i) continue;
            float4 pj4 = sim::PrecisionTraits::loadPosPred(sim::g_pos_pred, sim::g_pos_pred_h4, j);
            float3 pj  = make_float3(pj4.x, pj4.y, pj4.z);
            float3 rij = make_float3(pi.x - pj.x, pi.y - pj.y, pi.z - pj.z);
            float r2 = rij.x*rij.x + rij.y*rij.y + rij.z*rij.z;
            if (r2 > kc.h2) continue;
            neighborContrib++;
            float r = sqrtf(r2);
            float3 grad = spikyGrad(rij, r, kc);
            float lambda_j = sim::g_lambda[j];
            float w = (lambda_i + lambda_j);
            // 提供简单约束修正 (未考虑 sCorr / tensile)
            delta.x += w * grad.x;
            delta.y += w * grad.y;
            delta.z += w * grad.z;
        }
    }
Done:
    if (neighborContrib > 0 && restDensity > 0.f) {
        float scale = mass / restDensity;
        delta.x *= scale;
        delta.y *= scale;
        delta.z *= scale;
    }

    // XPBD compliance（简化：已在 lambda 中考虑，此处不再额外处理）
    // 应用位移
    pi.x += delta.x;
    pi.y += delta.y;
    pi.z += delta.z;

    sim::PrecisionTraits::storePosPred(sim::g_pos_pred, sim::g_pos_pred_h4, i, pi, pi4.w);
}

__global__ void KDeltaApplyGlobalsCompact(
    uint32_t N,
    const uint32_t* __restrict__ indicesSorted,
    const uint32_t* __restrict__ keysSorted,
    const uint32_t* __restrict__ uniqueKeys,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ compactCount,
    sim::DeviceParams dp)
{
    uint32_t sortedIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sortedIdx >= N) return;

    const auto& grid = dp.grid;
    const auto& kc   = dp.kernel;
    const float restDensity = dp.restDensity;
    const float mass        = dp.particleMass;
    const auto& pbf         = dp.pbf;

    const uint32_t M = *compactCount;
    uint32_t i = indicesSorted[sortedIdx];
    float4 pi4 = sim::PrecisionTraits::loadPosPred(sim::g_pos_pred, sim::g_pos_pred_h4, i);
    float3 pi  = make_float3(pi4.x, pi4.y, pi4.z);
    float lambda_i = sim::g_lambda[i];

    uint32_t key = keysSorted[sortedIdx];
    int3 ci;
    ci.x = int(key % uint32_t(grid.dim.x));
    uint32_t key_div_x = key / uint32_t(grid.dim.x);
    ci.y = int(key_div_x % uint32_t(grid.dim.y));
    ci.z = int(key_div_x / uint32_t(grid.dim.y));

    float3 delta = make_float3(0.f,0.f,0.f);
    int neighborContrib = 0;
    const int cap = dp.maxNeighbors;
    const bool hasCap = (cap > 0);
    const float cs = grid.cellSize;
    const int reach = (cs > 0.f) ? max(1, int(ceilf(kc.h / cs))) : 1;

    for (int dz=-reach; dz<=reach; ++dz)
    for (int dy=-reach; dy<=reach; ++dy)
    for (int dx=-reach; dx<=reach; ++dx) {
        if (hasCap && neighborContrib >= cap) goto Done;
        int3 cc = make_int3(ci.x+dx, ci.y+dy, ci.z+dz);
        if (!sim::inBounds(cc, grid.dim)) continue;
        uint32_t cidx = sim::linIdx(cc, grid.dim);

        uint32_t beg=0, end=0;
        if (!sim::compact_cell_range(uniqueKeys, offsets, M, cidx, beg, end)) continue;

        for (uint32_t k=beg; k<end; ++k) {
            if (hasCap && neighborContrib >= cap) break;
            uint32_t j = indicesSorted[k];
            if (j == i) continue;

            float4 pj4 = sim::PrecisionTraits::loadPosPred(sim::g_pos_pred, sim::g_pos_pred_h4, j);
            float3 pj  = make_float3(pj4.x, pj4.y, pj4.z);
            float3 rij = make_float3(pi.x - pj.x, pi.y - pj.y, pi.z - pj.z);
            float r2 = rij.x*rij.x + rij.y*rij.y + rij.z*rij.z;
            if (r2 > kc.h2) continue;
            neighborContrib++;
            float r = sqrtf(r2);
            float3 grad = spikyGrad(rij, r, kc);
            float lambda_j = sim::g_lambda[j];
            float w = (lambda_i + lambda_j);
            delta.x += w * grad.x;
            delta.y += w * grad.y;
            delta.z += w * grad.z;
        }
    }
Done:
    if (neighborContrib > 0 && restDensity > 0.f) {
        float scale = mass / restDensity;
        delta.x *= scale;
        delta.y *= scale;
        delta.z *= scale;
    }

    pi.x += delta.x;
    pi.y += delta.y;
    pi.z += delta.z;

    sim::PrecisionTraits::storePosPred(sim::g_pos_pred, sim::g_pos_pred_h4, i, pi, pi4.w);
}

} // namespace

extern "C" void LaunchDeltaApplyDenseGlobals(
    const uint32_t* indicesSorted,
    const uint32_t* keysSorted,
    const uint32_t* cellStart,
    const uint32_t* cellEnd,
    sim::DeviceParams dp,
    uint32_t N,
    cudaStream_t s)
{
    if (N==0) return;
    const int BS=256;
    dim3 block(BS), grid((N+BS-1)/BS);
    KDeltaApplyGlobalsDense<<<grid,block,0,s>>>(N, indicesSorted, keysSorted, cellStart, cellEnd, dp);
}

extern "C" void LaunchDeltaApplyCompactGlobals(
    const uint32_t* indicesSorted,
    const uint32_t* keysSorted,
    const uint32_t* uniqueKeys,
    const uint32_t* offsets,
    const uint32_t* compactCount,
    sim::DeviceParams dp,
    uint32_t N,
    cudaStream_t s)
{
    if (N==0) return;
    const int BS=256;
    dim3 block(BS), grid((N+BS-1)/BS);
    KDeltaApplyGlobalsCompact<<<grid,block,0,s>>>(N, indicesSorted, keysSorted, uniqueKeys, offsets, compactCount, dp);
}