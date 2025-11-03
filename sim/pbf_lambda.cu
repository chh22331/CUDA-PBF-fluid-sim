#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>
#include "parameters.h"
#include "cuda_vec_math.cuh"
#include "cuda_grid_utils.cuh"
#include "precision_traits.cuh" // 新增：统一读半/原生

namespace {

    __global__ void KLambda(float* lambda,
        const float4* pos_pred_fp32,
        const sim::Half4* pos_pred_h4,
        const uint32_t* indicesSorted,
        const uint32_t* keysSorted,
        const uint32_t* cellStart, const uint32_t* cellEnd,
        sim::DeviceParams dp, uint32_t N) {
        uint32_t sortedIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (sortedIdx >= N) return;

        uint32_t pid = indicesSorted[sortedIdx];
        uint32_t ghostCount = dp.ghostCount; uint32_t fluidCount = (ghostCount <= N)? (N - ghostCount): N; bool isGhost = (pid >= fluidCount);
        if (isGhost && !dp.ghostContribLambda) { lambda[pid]=0.f; return; }

        const sim::GridBounds& grid = dp.grid;
        const sim::KernelCoeffs& kc = dp.kernel;
        const float restDensity = dp.restDensity;
        const float particleMass = dp.particleMass;
        const sim::PbfTuning& pbf = dp.pbf;

        uint32_t i = indicesSorted[sortedIdx];
        float4 pi4 = sim::PrecisionTraits::loadPosPred(pos_pred_fp32, pos_pred_h4, i);
        float3 pi = make_float3(pi4.x, pi4.y, pi4.z);

        // 由 key 还原单元
        const uint32_t key = keysSorted[sortedIdx];
        int3 ci;
        ci.x = int(key % uint32_t(grid.dim.x));
        uint32_t key_div_x = key / uint32_t(grid.dim.x);
        ci.y = int(key_div_x % uint32_t(grid.dim.y));
        ci.z = int(key_div_x / uint32_t(grid.dim.y));

        float density = 0.f;
        float3 gradSum = make_float3(0.f, 0.f, 0.f);
        float sumGrad2 = 0.f;

        // 统计有效邻居（不含自项），用于“无邻居早退”
        int neighborContrib = 0;
        const int cap = dp.maxNeighbors;
        const bool hasCap = (cap > 0);
        const float h = kc.h;
        const float cs = grid.cellSize;
        const int reach = (cs > 0.f) ? max(1, int(ceilf(h / cs))) : 1;

        for (int dz = -reach; dz <= reach; ++dz) {
            for (int dy = -reach; dy <= reach; ++dy) {
                for (int dx = -reach; dx <= reach; ++dx) {
                    if (hasCap && neighborContrib >= cap) goto DoneNeighborsDense;

                    int3 cc = make_int3(ci.x + dx, ci.y + dy, ci.z + dz);
                    if (!sim::inBounds(cc, grid.dim)) continue;
                    uint32_t cidx = sim::linIdx(cc, grid.dim);
                    uint32_t beg = cellStart[cidx];
                    uint32_t end = cellEnd[cidx];
                    if (beg == 0xFFFFFFFFu || beg >= end) continue;

                    for (uint32_t k = beg; k < end; ++k) {
                        if (hasCap && neighborContrib >= cap) break;
                        uint32_t j = indicesSorted[k]; bool jGhost = (j >= fluidCount); if (jGhost && !dp.ghostContribLambda) continue;
                        if (j == i) continue;
                        float4 pj4 = sim::PrecisionTraits::loadPosPred(pos_pred_fp32, pos_pred_h4, j);
                        float3 pj = make_float3(pj4.x, pj4.y, pj4.z);
                        float3 rij = make_float3(pi.x - pj.x, pi.y - pj.y, pi.z - pj.z);
                        float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
                        if (r2 > kc.h2) continue;

                        neighborContrib++;

                        float hr2 = kc.h2 - r2;
                        float w = kc.poly6 * hr2 * hr2 * hr2;
                        density += particleMass * w;

                        float r = sqrtf(r2);
                        float r_safe = fmaxf(r, pbf.grad_r_eps);
                        float t = kc.h - r;
                        const float coeff = (-3.0f * kc.spiky * (t * t) / r_safe);
                        float3 grad = make_float3(coeff * rij.x, coeff * rij.y, coeff * rij.z);

                        gradSum.x += grad.x; gradSum.y += grad.y; gradSum.z += grad.z;
                        sumGrad2 += grad.x * grad.x + grad.y * grad.y + grad.z * grad.z;
                    }
                }
            }
        }
    DoneNeighborsDense:
        const float gradSum2 = gradSum.x * gradSum.x + gradSum.y * gradSum.y + gradSum.z * gradSum.z;
        const float scale = (restDensity > 0.f) ? (particleMass / restDensity) : 0.f;

        float denom = (scale * scale) * (sumGrad2 + gradSum2) + pbf.lambda_denom_eps;

        // XPBD：顺应性（0=关闭）
        if (pbf.compliance > 0.f && dp.dt > 0.f) {
            denom += pbf.compliance / (dp.dt * dp.dt);
        }
        denom = fmaxf(denom, pbf.lambda_denom_eps);

        // 邻居极少或梯度极小：早退，避免 lam 爆大
        if (neighborContrib < 2 || (sumGrad2 + gradSum2) < 1e-12f) {
            lambda[i] = 0.0f;
            return;
        }

        const float C = (restDensity > 0.f) ? (density / restDensity - 1.f) : 0.f;
        float lam = -C / denom;

        if (!isfinite(lam)) lam = 0.0f;
        if (pbf.enable_lambda_clamp) {
            const float a = fabsf(lam);
            if (a > pbf.lambda_max_abs) lam = copysignf(pbf.lambda_max_abs, lam);
        }

        lambda[i] = lam;
    }

    // —— 压缩段表版本 —— 
    __global__ void KLambdaCompact(
        float* __restrict__ lambda,
        const float4* __restrict__ pos_pred_fp32,
        const sim::Half4* __restrict__ pos_pred_h4,
        const uint32_t* __restrict__ indicesSorted,
        const uint32_t* __restrict__ keysSorted,
        const uint32_t* __restrict__ uniqueKeys,
        const uint32_t* __restrict__ offsets,
        const uint32_t* __restrict__ compactCount,
        sim::DeviceParams dp,
        uint32_t N)
    {
        uint32_t sortedIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (sortedIdx >= N) return;

        uint32_t pid = indicesSorted[sortedIdx]; uint32_t ghostCount = dp.ghostCount; uint32_t fluidCount = (ghostCount <= N)? (N - ghostCount): N; bool isGhost = (pid >= fluidCount); if (isGhost && !dp.ghostContribLambda){ lambda[pid]=0.f; return; }

        const sim::GridBounds& grid = dp.grid;
        const sim::KernelCoeffs& kc = dp.kernel;
        const float restDensity = dp.restDensity;
        const float particleMass = dp.particleMass;
        const sim::PbfTuning& pbf = dp.pbf;

        const uint32_t M = *compactCount;
        uint32_t i = indicesSorted[sortedIdx];
        float4 pi4 = sim::PrecisionTraits::loadPosPred(pos_pred_fp32, pos_pred_h4, i);
        float3 pi = make_float3(pi4.x, pi4.y, pi4.z);

        // 粒子自身所在单元（由 key 还原）
        const uint32_t key = keysSorted[sortedIdx];
        int3 ci;
        ci.x = int(key % uint32_t(grid.dim.x));
        uint32_t key_div_x = key / uint32_t(grid.dim.x);
        ci.y = int(key_div_x % uint32_t(grid.dim.y));
        ci.z = int(key_div_x / uint32_t(grid.dim.y));

        float density = 0.f;
        float3 gradSum = make_float3(0.f, 0.f, 0.f);
        float sumGrad2 = 0.f;

        int neighborContrib = 0;
        const int cap = dp.maxNeighbors;
        const bool hasCap = (cap > 0);
        const float h = kc.h;
        const float cs = grid.cellSize;
        const int reach = (cs > 0.f) ? max(1, int(ceilf(h / cs))) : 1;

        for (int dz = -reach; dz <= reach; ++dz) {
            for (int dy = -reach; dy <= reach; ++dy) {
                for (int dx = -reach; dx <= reach; ++dx) {
                    if (hasCap && neighborContrib >= cap) goto DoneNeighborsCompact;
                    int3 cc = make_int3(ci.x + dx, ci.y + dy, ci.z + dz);
                    if (!sim::inBounds(cc, grid.dim)) continue;
                    const uint32_t cidx = sim::linIdx(cc, grid.dim);

                    uint32_t beg = 0, end = 0;
                    if (!sim::compact_cell_range(uniqueKeys, offsets, M, cidx, beg, end)) continue;

                    for (uint32_t k = beg; k < end; ++k) {
                        if (hasCap && neighborContrib >= cap) break;
                        uint32_t j = indicesSorted[k]; bool jGhost = (j >= fluidCount); if (jGhost && !dp.ghostContribLambda) continue;
                        if (j == i) continue;

                        float4 pj4 = sim::PrecisionTraits::loadPosPred(pos_pred_fp32, pos_pred_h4, j);
                        float3 pj = make_float3(pj4.x, pj4.y, pj4.z);
                        float3 rij = make_float3(pi.x - pj.x, pi.y - pj.y, pi.z - pj.z);
                        float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
                        if (r2 > kc.h2) continue;

                        neighborContrib++;

                        float hr2 = kc.h2 - r2;
                        float w = kc.poly6 * hr2 * hr2 * hr2;
                        density += particleMass * w;

                        float r = sqrtf(r2);
                        float r_safe = fmaxf(r, pbf.grad_r_eps);
                        float t = kc.h - r;
                        const float coeff = (-3.0f * kc.spiky * (t * t) / r_safe);
                        float3 grad = make_float3(coeff * rij.x, coeff * rij.y, coeff * rij.z);

                        gradSum.x += grad.x; gradSum.y += grad.y; gradSum.z += grad.z;
                        sumGrad2 += grad.x * grad.x + grad.y * grad.y + grad.z * grad.z;
                    }
                }
            }
        }
    DoneNeighborsCompact:
        const float gradSum2 = gradSum.x * gradSum.x + gradSum.y * gradSum.y + gradSum.z * gradSum.z;
        const float scale = (restDensity > 0.f) ? (particleMass / restDensity) : 0.f;

        float denom = (scale * scale) * (sumGrad2 + gradSum2) + pbf.lambda_denom_eps;

        if (pbf.compliance > 0.f && dp.dt > 0.f) {
            denom += pbf.compliance / (dp.dt * dp.dt);
        }
        denom = fmaxf(denom, pbf.lambda_denom_eps);

        if (neighborContrib < 2 || (sumGrad2 + gradSum2) < 1e-12f) {
            lambda[i] = 0.0f;
            return;
        }

        const float C = (restDensity > 0.f) ? (density / restDensity - 1.f) : 0.f;
        float lam = -C / denom;

        if (!isfinite(lam)) lam = 0.0f;
        if (pbf.enable_lambda_clamp) {
            const float a = fabsf(lam);
            if (a > pbf.lambda_max_abs) lam = copysignf(pbf.lambda_max_abs, lam);
        }
        lambda[i] = lam;
    }

} // anon

extern "C" void LaunchLambda(float* lambda, const float4* pos_pred, const uint32_t* indicesSorted,
    const uint32_t* keysSorted,
    const uint32_t* cellStart, const uint32_t* cellEnd,
    sim::DeviceParams dp,
    uint32_t N, cudaStream_t s) {
    const int BS = 256;
    dim3 block(BS), gridDim((N + BS - 1) / BS);
    KLambda << <gridDim, block, 0, s >> > (lambda, pos_pred, nullptr, indicesSorted, keysSorted, cellStart, cellEnd, dp, N);
}

extern "C" void LaunchLambdaCompact(
    float* lambda,
    const float4* pos_pred,
    const uint32_t* indicesSorted,
    const uint32_t* keysSorted,
    const uint32_t* uniqueKeys,
    const uint32_t* offsets,
    const uint32_t* compactCount,
    sim::DeviceParams dp,
    uint32_t N,
    cudaStream_t s)
{
    const int BS = 256;
    dim3 block(BS), gridDim((N + BS - 1) / BS);
    KLambdaCompact<<<gridDim, block, 0, s>>>(
        lambda, pos_pred, nullptr, indicesSorted, keysSorted,
        uniqueKeys, offsets, compactCount, dp, N);
}