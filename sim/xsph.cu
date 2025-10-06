#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "parameters.h"
#include "cuda_vec_math.cuh"
#include "cuda_grid_utils.cuh"

namespace {

    // XSPH: v_i' = v_i + c * (m/ρ0) * Σ_j (v_j - v_i) W_poly6(x_i - x_j)
    __global__ void KXSPH(
        float4* __restrict__ vel_out,
        const float4* __restrict__ vel_in,
        const float4* __restrict__ pos_pred,
        const uint32_t* __restrict__ indicesSorted,
        const uint32_t* __restrict__ cellStart,
        const uint32_t* __restrict__ cellEnd,
        sim::DeviceParams dp,
        uint32_t N)
    {
        uint32_t sortedIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (sortedIdx >= N) return;

        const sim::GridBounds& grid = dp.grid;
        const sim::KernelCoeffs& kc = dp.kernel;
        const float xsph_c = dp.xsph_c;

        uint32_t i = indicesSorted[sortedIdx];
        float3 pi = to_float3(pos_pred[i]);
        float3 vi = to_float3(vel_in[i]);

        float3 rel = make_float3((pi.x - grid.mins.x) / grid.cellSize,
            (pi.y - grid.mins.y) / grid.cellSize,
            (pi.z - grid.mins.z) / grid.cellSize);
        int3 ci = make_int3(floorf(rel.x), floorf(rel.y), floorf(rel.z));
        ci.x = max(0, min(ci.x, grid.dim.x - 1));
        ci.y = max(0, min(ci.y, grid.dim.y - 1));
        ci.z = max(0, min(ci.z, grid.dim.z - 1));

        float3 dv_sum = make_float3(0.f, 0.f, 0.f);

        const bool capEnabled = (dp.maxNeighbors > 0);
        int neighborCount = 0;

        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (capEnabled && neighborCount >= dp.maxNeighbors) break;

                    int3 cc = make_int3(ci.x + dx, ci.y + dy, ci.z + dz);
                    if (cc.x < 0 || cc.x >= grid.dim.x ||
                        cc.y < 0 || cc.y >= grid.dim.y ||
                        cc.z < 0 || cc.z >= grid.dim.z) {
                        continue;
                    }

                    uint32_t cidx = static_cast<uint32_t>((cc.z * grid.dim.y + cc.y) * grid.dim.x + cc.x);
                    uint32_t beg = cellStart[cidx];
                    uint32_t end = cellEnd[cidx];
                    if (beg == 0xFFFFFFFFu || beg >= end) continue;

                    for (uint32_t k = beg; k < end; ++k) {
                        if (capEnabled && neighborCount >= dp.maxNeighbors) break;

                        uint32_t j = indicesSorted[k];
                        if (j == i) continue;

                        float3 pj = to_float3(pos_pred[j]);
                        float3 rij = make_float3(pi.x - pj.x, pi.y - pj.y, pi.z - pj.z);
                        float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
                        if (r2 > kc.h2) continue;

                        float hr2 = kc.h2 - r2;
                        float w = kc.poly6 * hr2 * hr2 * hr2;

                        float3 vj = to_float3(vel_in[j]);
                        dv_sum.x += (vj.x - vi.x) * w;
                        dv_sum.y += (vj.y - vi.y) * w;
                        dv_sum.z += (vj.z - vi.z) * w;

                        ++neighborCount;
                    }
                }
            }
        }

        const float mass_over_rest = (dp.restDensity > 0.f) ? (dp.particleMass / dp.restDensity) : 0.f;
        float3 vnew = make_float3(
            vi.x + xsph_c * mass_over_rest * dv_sum.x,
            vi.y + xsph_c * mass_over_rest * dv_sum.y,
            vi.z + xsph_c * mass_over_rest * dv_sum.z
        );

        vel_out[i] = make_float4(vnew.x, vnew.y, vnew.z, 0.0f);
    }

} // anon

extern "C" void LaunchXSPH(
    float4* vel_out,
    const float4* vel_in,
    const float4* pos_pred,
    const uint32_t* indicesSorted,
    const uint32_t* cellStart,
    const uint32_t* cellEnd,
    sim::DeviceParams dp,
    uint32_t N,
    cudaStream_t s)
{
    if (N == 0 || dp.xsph_c <= 0.f) return;
    const int BS = 256;
    dim3 block(BS), gridDim((N + BS - 1) / BS);
    KXSPH << <gridDim, block, 0, s >> > (vel_out, vel_in, pos_pred, indicesSorted, cellStart, cellEnd, dp, N);
}