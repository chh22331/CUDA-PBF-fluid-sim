#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "parameters.h"
#include "cuda_vec_math.cuh"
#include "cuda_grid_utils.cuh"
#include "precision_traits.cuh"

namespace {

    __global__ void KDeltaCompute(
        float4* __restrict__ delta,
        const float4* __restrict__ pos_pred_fp32,
        const sim::Half4* __restrict__ pos_pred_h4,
        const float* __restrict__ lambda,
        const uint32_t* __restrict__ indicesSorted,
        const uint32_t* __restrict__ keysSorted,
        const uint32_t* __restrict__ cellStart,
        const uint32_t* __restrict__ cellEnd,
        sim::DeviceParams dp,
        uint32_t N)
    {
        uint32_t sortedIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (sortedIdx >= N) return;

        const sim::GridBounds& grid = dp.grid;
        const sim::KernelCoeffs& kc = dp.kernel;
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

        const float dq = pbf.scorr_dq_h * kc.h;
        const float hr2_q = kc.h2 - dq * dq;
        const float w_q = (hr2_q > 0.f) ? (kc.poly6 * hr2_q * hr2_q * hr2_q) : pbf.wq_min;

        float3 dxi = make_float3(0.f, 0.f, 0.f);

        const float h = kc.h;
        const float cs = grid.cellSize;
        const int reach = (cs > 0.f) ? max(1, int(ceilf(h / cs))) : 1;

        for (int dz = -reach; dz <= reach; ++dz) {
            for (int dy = -reach; dy <= reach; ++dy) {
                for (int dx = -reach; dx <= reach; ++dx) {
                    int3 cc = make_int3(ci.x + dx, ci.y + dy, ci.z + dz);
                    if (!sim::inBounds(cc, grid.dim)) continue;

                    uint32_t cidx = sim::linIdx(cc, grid.dim);
                    uint32_t beg = cellStart[cidx];
                    uint32_t end = cellEnd[cidx];
                    if (beg == 0xFFFFFFFFu || beg >= end) continue;

                    for (uint32_t k = beg; k < end; ++k) {
                        uint32_t j = indicesSorted[k];
                        if (j == i) continue;

                        float4 pj4 = sim::PrecisionTraits::loadPosPred(pos_pred_fp32, pos_pred_h4, j);
                        float3 pj = make_float3(pj4.x, pj4.y, pj4.z);
                        float3 rij = make_float3(pi.x - pj.x, pi.y - pj.y, pi.z - pj.z);
                        float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
                        if (r2 > kc.h2) continue;

                        float r = sqrtf(r2);
                        float t = kc.h - r;

                        // 与 λ 阶段一致的 r_safe
                        const float r_safe = fmaxf(r, pbf.grad_r_eps);
                        const float coeff = (-3.0f * kc.spiky * (t * t) / r_safe);
                        float3 grad = make_float3(coeff * rij.x, coeff * rij.y, coeff * rij.z);

                        float scorr = 0.0f;
                        if (pbf.scorr_enable) {
                            float hr2 = kc.h2 - r2;
                            float w = kc.poly6 * hr2 * hr2 * hr2;
                            const float ratio = (w_q > 0.f) ? (w / w_q) : 0.0f;
                            scorr = -pbf.scorr_k * powf(ratio, pbf.scorr_n);
                            if (pbf.scorr_min < 0.f) scorr = fmaxf(scorr, pbf.scorr_min);
                        }

                        const float lij = (lambda[i] + lambda[j] + scorr);
                        dxi.x += lij * grad.x;
                        dxi.y += lij * grad.y;
                        dxi.z += lij * grad.z;
                    }
                }
            }
        }

        // 修正：使用 1/ρ0（标准 PBF），而非 m/ρ0
        const float invRest = (dp.restDensity > 0.f) ? (1.0f / dp.restDensity) : 0.f;
        float3 di = make_float3(invRest * dxi.x, invRest * dxi.y, invRest * dxi.z);

        if (pbf.enable_relax) {
            di.x *= pbf.relax_omega;
            di.y *= pbf.relax_omega;
            di.z *= pbf.relax_omega;
        }

        if (pbf.enable_disp_clamp) {
            const float maxDisp = pbf.disp_clamp_max_h * kc.h;
            const float maxDisp2 = maxDisp * maxDisp;
            const float len2 = di.x * di.x + di.y * di.y + di.z * di.z;
            if (len2 > maxDisp2) {
                const float len = sqrtf(len2);
                const float s = maxDisp / (len + 1e-20f);
                di.x *= s; di.y *= s; di.z *= s;
            }
        }

        delta[i] = make_float4(di.x, di.y, di.z, 0.0f);
    }

    __global__ void KDeltaComputeCompact(
        float4* __restrict__ delta,
        const float4* __restrict__ pos_pred_fp32,
        const sim::Half4* __restrict__ pos_pred_h4,
        const float* __restrict__ lambda,
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

        const sim::GridBounds& grid = dp.grid;
        const sim::KernelCoeffs& kc = dp.kernel;
        const sim::PbfTuning& pbf = dp.pbf;

        const uint32_t M = *compactCount;

        uint32_t i = indicesSorted[sortedIdx];
        float4 pi4 = sim::PrecisionTraits::loadPosPred(pos_pred_fp32, pos_pred_h4, i);
        float3 pi = make_float3(pi4.x, pi4.y, pi4.z);

        const uint32_t key = keysSorted[sortedIdx];
        int3 ci;
        ci.x = int(key % uint32_t(grid.dim.x));
        uint32_t key_div_x = key / uint32_t(grid.dim.x);
        ci.y = int(key_div_x % uint32_t(grid.dim.y));
        ci.z = int(key_div_x / uint32_t(grid.dim.y));

        const float dq = pbf.scorr_dq_h * kc.h;
        const float hr2_q = kc.h2 - dq * dq;
        const float w_q = (hr2_q > 0.f) ? (kc.poly6 * hr2_q * hr2_q * hr2_q) : pbf.wq_min;

        float3 dxi = make_float3(0.f, 0.f, 0.f);

        const float h = kc.h;
        const float cs = grid.cellSize;
        const int reach = (cs > 0.f) ? max(1, int(ceilf(h / cs))) : 1;

        for (int dz = -reach; dz <= reach; ++dz) {
            for (int dy = -reach; dy <= reach; ++dy) {
                for (int dx = -reach; dx <= reach; ++dx) {
                    int3 cc = make_int3(ci.x + dx, ci.y + dy, ci.z + dz);
                    if (!sim::inBounds(cc, grid.dim)) continue;

                    const uint32_t cidx = sim::linIdx(cc, grid.dim);
                    uint32_t beg = 0, end = 0;
                    if (!sim::compact_cell_range(uniqueKeys, offsets, M, cidx, beg, end)) continue;

                    for (uint32_t k = beg; k < end; ++k) {
                        uint32_t j = indicesSorted[k];
                        if (j == i) continue;

                        float4 pj4 = sim::PrecisionTraits::loadPosPred(pos_pred_fp32, pos_pred_h4, j);
                        float3 pj = make_float3(pj4.x, pj4.y, pj4.z);
                        float3 rij = make_float3(pi.x - pj.x, pi.y - pj.y, pi.z - pj.z);
                        float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
                        if (r2 > kc.h2) continue;

                        float r = sqrtf(r2);
                        float t = kc.h - r;

                        const float r_safe = fmaxf(r, pbf.grad_r_eps);
                        const float coeff = (-3.0f * kc.spiky * (t * t) / r_safe);
                        float3 grad = make_float3(coeff * rij.x, coeff * rij.y, coeff * rij.z);

                        float scorr = 0.0f;
                        if (pbf.scorr_enable) {
                            float hr2 = kc.h2 - r2;
                            float w = kc.poly6 * hr2 * hr2 * hr2;
                            const float ratio = (w_q > 0.f) ? (w / w_q) : 0.0f;
                            scorr = -pbf.scorr_k * powf(ratio, pbf.scorr_n);
                            if (pbf.scorr_min < 0.f) scorr = fmaxf(scorr, pbf.scorr_min);
                        }

                        const float lij = (lambda[i] + lambda[j] + scorr);
                        dxi.x += lij * grad.x;
                        dxi.y += lij * grad.y;
                        dxi.z += lij * grad.z;
                    }
                }
            }
        }

        const float invRest = (dp.restDensity > 0.f) ? (1.0f / dp.restDensity) : 0.f;
        float3 di = make_float3(invRest * dxi.x, invRest * dxi.y, invRest * dxi.z);

        if (pbf.enable_relax) {
            di.x *= pbf.relax_omega;
            di.y *= pbf.relax_omega;
            di.z *= pbf.relax_omega;
        }
        if (pbf.enable_disp_clamp) {
            const float maxDisp = pbf.disp_clamp_max_h * kc.h;
            const float maxDisp2 = maxDisp * maxDisp;
            const float len2 = di.x * di.x + di.y * di.y + di.z * di.z;
            if (len2 > maxDisp2) {
                const float len = sqrtf(len2);
                const float s = maxDisp / (len + 1e-20f);
                di.x *= s; di.y *= s; di.z *= s;
            }
        }

        delta[i] = make_float4(di.x, di.y, di.z, 0.0f);
    }

    __global__ void KApplyDelta(
        float4* __restrict__ pos_pred_fp32,
        sim::Half4* __restrict__ pos_pred_h4,
        const float4* __restrict__ delta,
        const uint32_t* __restrict__ indicesSorted,
        uint32_t N)
    {
        uint32_t sortedIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (sortedIdx >= N) return;

        uint32_t i = indicesSorted[sortedIdx];
        float4 di4 = delta[i];
        float4 base = sim::PrecisionTraits::loadPosPred(pos_pred_fp32, pos_pred_h4, i);
        float3 newP = make_float3(base.x + di4.x, base.y + di4.y, base.z + di4.z);
        sim::PrecisionTraits::storePosPred(pos_pred_fp32, pos_pred_h4, i, newP, 1.0f);
    }

} // anon

extern "C" void LaunchDeltaApply(
    float4* pos_pred,
    float4* delta,
    const float* lambda,
    const uint32_t* indicesSorted,
    const uint32_t* keysSorted,
    const uint32_t* cellStart,
    const uint32_t* cellEnd,
    sim::DeviceParams dp,
    uint32_t N,
    cudaStream_t s)
{
    const int BS = 256;
    dim3 block(BS), gridDim((N + BS - 1) / BS);

    KDeltaCompute<<<gridDim, block, 0, s>>>(
        delta, pos_pred, nullptr, lambda, indicesSorted, keysSorted, cellStart, cellEnd, dp, N);
    KApplyDelta<<<gridDim, block, 0, s>>>(pos_pred, nullptr, delta, indicesSorted, N);
}

extern "C" void LaunchDeltaApplyCompact(
    float4* pos_pred,
    float4* delta,
    const float* lambda,
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

    KDeltaComputeCompact<<<gridDim, block, 0, s>>>(
        delta, pos_pred, nullptr, lambda, indicesSorted, keysSorted, uniqueKeys, offsets, compactCount, dp, N);
    KApplyDelta<<<gridDim, block, 0, s>>>(pos_pred, nullptr, delta, indicesSorted, N);
}