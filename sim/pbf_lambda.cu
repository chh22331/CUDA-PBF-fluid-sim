#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "parameters.h"
#include "cuda_vec_math.cuh"
#include "cuda_grid_utils.cuh"

namespace {

    __global__ void KLambda(float* lambda, const float4* pos_pred,
        const uint32_t* indicesSorted,
        const uint32_t* cellStart, const uint32_t* cellEnd,
        sim::DeviceParams dp, uint32_t N) {
        uint32_t sortedIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (sortedIdx >= N) return;

        const sim::GridBounds& grid = dp.grid;
        const sim::KernelCoeffs& kc = dp.kernel;
        const float restDensity = dp.restDensity;
        const float particleMass = dp.particleMass;
        const sim::PbfTuning& pbf = dp.pbf;

        uint32_t i = indicesSorted[sortedIdx];
        float3 pi = to_float3(pos_pred[i]);

        float3 rel = make_float3((pi.x - grid.mins.x) / grid.cellSize,
                                 (pi.y - grid.mins.y) / grid.cellSize,
                                 (pi.z - grid.mins.z) / grid.cellSize);
        int3 ci = make_int3(floorf(rel.x), floorf(rel.y), floorf(rel.z));
        // 钳制锚单元，避免落在 dim 之外导致大面积漏邻居
        ci.x = max(0, min(ci.x, grid.dim.x - 1));
        ci.y = max(0, min(ci.y, grid.dim.y - 1));
        ci.z = max(0, min(ci.z, grid.dim.z - 1));

        float density = 0.f;
        float3 gradSum = make_float3(0.f, 0.f, 0.f);
        float sumGrad2 = 0.f;

        int neighborCount = 0;
        const int maxN = dp.maxNeighbors > 0 ? dp.maxNeighbors : INT_MAX;

        // 自项密度：j==i, 梯度自项为 0
        {
            const float hr2 = kc.h2; // r=0 => h^2 - 0
            const float w0 = kc.poly6 * hr2 * hr2 * hr2;
            density += particleMass * w0;
        }

        for (int dz = -1; dz <= 1; ++dz) {
            if (neighborCount >= maxN) break;
            for (int dy = -1; dy <= 1; ++dy) {
                if (neighborCount >= maxN) break;
                for (int dx = -1; dx <= 1; ++dx) {
                    if (neighborCount >= maxN) break;

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
                        if (neighborCount >= maxN) break;

                        uint32_t j = indicesSorted[k];
                        if (j == i) continue;
                        float3 pj = to_float3(pos_pred[j]);
                        float3 rij = make_float3(pi.x - pj.x, pi.y - pj.y, pi.z - pj.z);
                        float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
                        if (r2 > kc.h2) continue;

                        // ρ_i += m_j W_poly6
                        float hr2 = kc.h2 - r2;
                        float w = kc.poly6 * hr2 * hr2 * hr2;
                        density += particleMass * w;

                        // ∇_xi W_spiky
                        float r = sqrtf(r2);
                        float t = kc.h - r;
                        const float coeff = (r > pbf.grad_r_eps) ? (-3.0f * kc.spiky * (t * t) / r) : 0.0f;
                        float3 grad = make_float3(coeff * rij.x, coeff * rij.y, coeff * rij.z);

                        gradSum.x += grad.x; gradSum.y += grad.y; gradSum.z += grad.z;
                        sumGrad2 += grad.x * grad.x + grad.y * grad.y + grad.z * grad.z;

                        ++neighborCount;
                    }
                }
            }
        }

        // denom = (m/ρ0)^2 [ Σ|∇W|^2 + |Σ∇W|^2 ] + eps
        const float gradSum2 = gradSum.x * gradSum.x + gradSum.y * gradSum.y + gradSum.z * gradSum.z;
        const float scale = (restDensity > 0.f) ? (particleMass / restDensity) : 0.f;
        const float denom = (scale * scale) * (sumGrad2 + gradSum2) + pbf.lambda_denom_eps;

        const float C = density / restDensity - 1.f;
        lambda[i] = -C / denom;
    }

} // anon

extern "C" void LaunchLambda(float* lambda, const float4* pos_pred, const uint32_t* indicesSorted,
    const uint32_t* cellStart, const uint32_t* cellEnd,
    sim::DeviceParams dp,
    uint32_t N, cudaStream_t s) {
    const int BS = 256;
    dim3 block(BS), gridDim((N + BS - 1) / BS);
    KLambda<<<gridDim, block, 0, s>>>(lambda, pos_pred, indicesSorted, cellStart, cellEnd, dp, N);
}