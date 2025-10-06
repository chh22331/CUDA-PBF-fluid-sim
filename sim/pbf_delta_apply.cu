#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "parameters.h"
#include "cuda_vec_math.cuh"
#include "cuda_grid_utils.cuh"

namespace {

    // 仅计算 delta[i]，不写 pos_pred，避免读写竞态
    __global__ void KDeltaCompute(
        float4* __restrict__ delta,
        const float4* __restrict__ pos_pred,
        const float* __restrict__ lambda,
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
        const sim::PbfTuning& pbf = dp.pbf;

        uint32_t i = indicesSorted[sortedIdx];
        float3 pi = to_float3(pos_pred[i]);

        float3 rel = make_float3((pi.x - grid.mins.x) / grid.cellSize,
                                 (pi.y - grid.mins.y) / grid.cellSize,
                                 (pi.z - grid.mins.z) / grid.cellSize);
        int3 ci = make_int3(floorf(rel.x), floorf(rel.y), floorf(rel.z));
        ci.x = max(0, min(ci.x, grid.dim.x - 1));
        ci.y = max(0, min(ci.y, grid.dim.y - 1));
        ci.z = max(0, min(ci.z, grid.dim.z - 1));

        // s_corr 参考权重（poly6）
        const float dq = pbf.scorr_dq_h * kc.h;
        const float hr2_q = kc.h2 - dq * dq;
        const float w_q = (hr2_q > 0.f) ? (kc.poly6 * hr2_q * hr2_q * hr2_q) : pbf.wq_min;

        float3 dxi = make_float3(0.f, 0.f, 0.f);

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

                        float r = sqrtf(r2);
                        float t = kc.h - r;

                        // ∇_xi W_spiky = 15/(π h^6) * (h - r)^2 * ( -rij / r )
                        const float coeff = (r > pbf.grad_r_eps) ? (-3.0f * kc.spiky * (t * t) / r) : 0.0f;
                        float3 grad = make_float3(coeff * rij.x, coeff * rij.y, coeff * rij.z);

                        // s_corr（负号排斥）
                        float hr2 = kc.h2 - r2;
                        float w = kc.poly6 * hr2 * hr2 * hr2;
                        float scorr = -pbf.scorr_k * powf(w / w_q, pbf.scorr_n);

                        // 标准 PBF：Δx_i 累加核（尚未乘以 m/ρ0）
                        float lij = (lambda[i] + lambda[j] + scorr);
                        dxi.x += lij * grad.x;
                        dxi.y += lij * grad.y;
                        dxi.z += lij * grad.z;

                        ++neighborCount;
                    }
                }
            }
        }

        // 按质量与ρ0缩放（缺失这一项会放大约束位移 ~1/m 倍）
        const float mass_over_rest = (dp.restDensity > 0.f) ? (dp.particleMass / dp.restDensity) : 0.f;
        float3 di = make_float3(mass_over_rest * dxi.x, mass_over_rest * dxi.y, mass_over_rest * dxi.z);

        // 可选：位移限幅，避免一次迭代过大（数值更稳）
        const float maxDisp = 0.5f * kc.h;
        const float len2 = di.x * di.x + di.y * di.y + di.z * di.z;
        if (len2 > maxDisp * maxDisp) {
            const float len = sqrtf(len2);
            const float s = maxDisp / (len + 1e-20f);
            di.x *= s; di.y *= s; di.z *= s;
        }

        delta[i] = make_float4(di.x, di.y, di.z, 0.0f);
    }

    // 将预计算的 delta 应用到 pos_pred
    __global__ void KApplyDelta(float4* __restrict__ pos_pred,
                                const float4* __restrict__ delta,
                                const uint32_t* __restrict__ indicesSorted,
                                uint32_t N)
    {
        uint32_t sortedIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (sortedIdx >= N) return;

        uint32_t i = indicesSorted[sortedIdx];
        float4 pi4 = pos_pred[i];
        float4 di4 = delta[i];
        pos_pred[i] = make_float4(pi4.x + di4.x, pi4.y + di4.y, pi4.z + di4.z, 1.0f);
    }

} // anon

extern "C" void LaunchDeltaApply(
    float4* pos_pred,
    float4* delta,
    const float* lambda,
    const uint32_t* indicesSorted,
    const uint32_t* cellStart,
    const uint32_t* cellEnd,
    sim::DeviceParams dp,
    uint32_t N,
    cudaStream_t s)
{
    const int BS = 256;
    dim3 block(BS), gridDim((N + BS - 1) / BS);

    // 先计算 delta（只读 pos_pred）
    KDeltaCompute<<<gridDim, block, 0, s>>>(delta, pos_pred, lambda, indicesSorted, cellStart, cellEnd, dp, N);
    // 再统一应用
    KApplyDelta<<<gridDim, block, 0, s>>>(pos_pred, delta, indicesSorted, N);
}