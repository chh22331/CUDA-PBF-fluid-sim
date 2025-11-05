// 修改：移除对 g_vel 的直接写入，增加权重归一化
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "device_globals.cuh"
#include "cuda_vec_math.cuh"
#include "cuda_grid_utils.cuh"
#include "precision_traits.cuh"
#include "parameters.h"

namespace {
__global__ void KXsphGlobalsDense(
    uint32_t N,
    const uint32_t* __restrict__ indicesSorted,
    const uint32_t* __restrict__ keysSorted,
    const uint32_t* __restrict__ cellStart,
    const uint32_t* __restrict__ cellEnd,
    sim::DeviceParams dp)
{
    uint32_t sortedIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sortedIdx >= N || dp.xsph_c <= 0.f) return;

    const auto& grid = dp.grid;
    const auto& kc   = dp.kernel;
    float c          = dp.xsph_c;

    uint32_t i = indicesSorted[sortedIdx];
    float4 pi4 = sim::g_pos_pred[i];
    float3 vi  = make_float3(sim::g_vel[i].x, sim::g_vel[i].y, sim::g_vel[i].z);

    uint32_t key = keysSorted[sortedIdx];
    int3 ci;
    ci.x = int(key % grid.dim.x);
    uint32_t key_div_x = key / grid.dim.x;
    ci.y = int(key_div_x % grid.dim.y);
    ci.z = int(key_div_x / grid.dim.y);

    float3 sum = make_float3(0.f,0.f,0.f);
    float  weightSum = 0.f;
    int neighborContrib = 0;
    const int cap = dp.maxNeighbors;
    const bool hasCap = (cap>0);
    const float cs = grid.cellSize;
    const int reach = (cs>0.f)? max(1,int(ceilf(kc.h / cs))):1;

    for (int dz=-reach; dz<=reach; ++dz)
    for (int dy=-reach; dy<=reach; ++dy)
    for (int dx=-reach; dx<=reach; ++dx) {
        if (hasCap && neighborContrib>=cap) goto Done;
        int3 cc = make_int3(ci.x+dx, ci.y+dy, ci.z+dz);
        if (!sim::inBounds(cc, grid.dim)) continue;
        uint32_t cidx = sim::linIdx(cc, grid.dim);
        uint32_t beg = cellStart[cidx];
        uint32_t end = cellEnd[cidx];
        if (beg == 0xFFFFFFFFu || beg >= end) continue;

        for (uint32_t k=beg; k<end; ++k) {
            if (hasCap && neighborContrib>=cap) break;
            uint32_t j = indicesSorted[k];
            if (j == i) continue;
            float4 pj4 = sim::g_pos_pred[j];
            float3 rij = make_float3(pi4.x - pj4.x, pi4.y - pj4.y, pi4.z - pj4.z);
            float r2 = rij.x*rij.x + rij.y*rij.y + rij.z*rij.z;
            if (r2 > kc.h2) continue;
            neighborContrib++;
            float hr2 = kc.h2 - r2;
            float w = kc.poly6 * hr2 * hr2 * hr2;
            float3 vj = make_float3(sim::g_vel[j].x, sim::g_vel[j].y, sim::g_vel[j].z);
            sum.x += (vj.x - vi.x)*w;
            sum.y += (vj.y - vi.y)*w;
            sum.z += (vj.z - vi.z)*w;
            weightSum += w;
        }
    }
Done:
    if (weightSum > 1e-8f) {
        sum.x /= weightSum;
        sum.y /= weightSum;
        sum.z /= weightSum;
    }
    float3 vout = make_float3(vi.x + c*sum.x,
                              vi.y + c*sum.y,
                              vi.z + c*sum.z);

    // 仅写入平滑结果到 g_delta，延后在 Boundary 阶段提交到 g_vel
    sim::g_delta[i] = make_float4(vout.x,vout.y,vout.z,0.f);
}

__global__ void KXsphGlobalsCompact(
    uint32_t N,
    const uint32_t* __restrict__ indicesSorted,
    const uint32_t* __restrict__ keysSorted,
    const uint32_t* __restrict__ uniqueKeys,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ compactCount,
    sim::DeviceParams dp)
{
    uint32_t sortedIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sortedIdx >= N || dp.xsph_c <= 0.f) return;

    const auto& grid = dp.grid;
    const auto& kc   = dp.kernel;
    float c          = dp.xsph_c;

    const uint32_t M = *compactCount;
    uint32_t i = indicesSorted[sortedIdx];
    float4 pi4 = sim::g_pos_pred[i];
    float3 vi  = make_float3(sim::g_vel[i].x, sim::g_vel[i].y, sim::g_vel[i].z);

    uint32_t key = keysSorted[sortedIdx];
    int3 ci;
    ci.x = int(key % grid.dim.x);
    uint32_t key_div_x = key / grid.dim.x;
    ci.y = int(key_div_x % grid.dim.y);
    ci.z = int(key_div_x / grid.dim.y);

    float3 sum = make_float3(0.f,0.f,0.f);
    float  weightSum = 0.f;
    int neighborContrib=0;
    const int cap = dp.maxNeighbors;
    const bool hasCap = (cap>0);
    const float cs = grid.cellSize;
    const int reach = (cs>0.f)? max(1,int(ceilf(kc.h / cs))):1;

    for (int dz=-reach; dz<=reach; ++dz)
    for (int dy=-reach; dy<=reach; ++dy)
    for (int dx=-reach; dx<=reach; ++dx) {
        if (hasCap && neighborContrib>=cap) goto Done;
        int3 cc = make_int3(ci.x+dx, ci.y+dy, ci.z+dz);
        if (!sim::inBounds(cc, grid.dim)) continue;
        uint32_t cidx = sim::linIdx(cc, grid.dim);

        uint32_t beg=0,end=0;
        if (!sim::compact_cell_range(uniqueKeys, offsets, M, cidx, beg, end)) continue;

        for (uint32_t k=beg; k<end; ++k) {
            if (hasCap && neighborContrib>=cap) break;
            uint32_t j = indicesSorted[k];
            if (j == i) continue;
            float4 pj4 = sim::g_pos_pred[j];
            float3 rij = make_float3(pi4.x - pj4.x, pi4.y - pj4.y, pi4.z - pj4.z);
            float r2 = rij.x*rij.x + rij.y*rij.y + rij.z*rij.z;
            if (r2 > kc.h2) continue;
            neighborContrib++;
            float hr2 = kc.h2 - r2;
            float w = kc.poly6 * hr2 * hr2 * hr2;
            float3 vj = make_float3(sim::g_vel[j].x, sim::g_vel[j].y, sim::g_vel[j].z);
            sum.x += (vj.x - vi.x)*w;
            sum.y += (vj.y - vi.y)*w;
            sum.z += (vj.z - vi.z)*w;
            weightSum += w;
        }
    }
Done:
    if (weightSum > 1e-8f) {
        sum.x /= weightSum;
        sum.y /= weightSum;
        sum.z /= weightSum;
    }
    float3 vout = make_float3(vi.x + c*sum.x,
                              vi.y + c*sum.y,
                              vi.z + c*sum.z);

    sim::g_delta[i] = make_float4(vout.x,vout.y,vout.z,0.f);
}
} // namespace

extern "C" void LaunchXsphDenseGlobals(
    const uint32_t* indicesSorted,
    const uint32_t* keysSorted,
    const uint32_t* cellStart,
    const uint32_t* cellEnd,
    sim::DeviceParams dp,
    uint32_t N,
    cudaStream_t s)
{
    if (N==0 || dp.xsph_c<=0.f) return;
    const int BS=256; dim3 block(BS), grid((N+BS-1)/BS);
    KXsphGlobalsDense<<<grid,block,0,s>>>(N, indicesSorted, keysSorted, cellStart, cellEnd, dp);
}

extern "C" void LaunchXsphCompactGlobals(
    const uint32_t* indicesSorted,
    const uint32_t* keysSorted,
    const uint32_t* uniqueKeys,
    const uint32_t* offsets,
    const uint32_t* compactCount,
    sim::DeviceParams dp,
    uint32_t N,
    cudaStream_t s)
{
    if (N==0 || dp.xsph_c<=0.f) return;
    const int BS=256; dim3 block(BS), grid((N+BS-1)/BS);
    KXsphGlobalsCompact<<<grid,block,0,s>>>(N, indicesSorted, keysSorted, uniqueKeys, offsets, compactCount, dp);
}