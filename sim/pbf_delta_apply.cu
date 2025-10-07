#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "parameters.h"
#include "cuda_vec_math.cuh"
#include "cuda_grid_utils.cuh"

namespace {

    __global__ void KDeltaCompute(
        float4* __restrict__ delta,
        const float4* __restrict__ pos_pred,
        const float* __restrict__ lambda,
        const uint32_t* __restrict__ indicesSorted,
        const uint32_t* __restrict__ keysSorted,        // 新增
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

        // 由 key 还原单元（固定为建表时的单元）
        const uint32_t key = keysSorted[sortedIdx];
        int3 ci;
        ci.x = int(key % uint32_t(grid.dim.x));
        uint32_t key_div_x = key / uint32_t(grid.dim.x);
        ci.y = int(key_div_x % uint32_t(grid.dim.y));
        ci.z = int(key_div_x / uint32_t(grid.dim.y));

        // s_corr 参考权重
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

                        float3 pj = to_float3(pos_pred[j]);
                        float3 rij = make_float3(pi.x - pj.x, pi.y - pj.y, pi.z - pj.z);
                        float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
                        if (r2 > kc.h2) continue;

                        float r = sqrtf(r2);
                        float t = kc.h - r;

                        const float coeff = (r > pbf.grad_r_eps) ? (-3.0f * kc.spiky * (t * t) / r) : 0.0f;
                        float3 grad = make_float3(coeff * rij.x, coeff * rij.y, coeff * rij.z);

                        float scorr = 0.0f;
                        if (pbf.scorr_enable) {
                            float hr2 = kc.h2 - r2;
                            float w = kc.poly6 * hr2 * hr2 * hr2;
                            const float ratio = (w_q > 0.f) ? (w / w_q) : 0.0f;
                            scorr = -pbf.scorr_k * powf(ratio, pbf.scorr_n);
                        }

                        const float lij = (lambda[i] + lambda[j] + scorr);
                        dxi.x += lij * grad.x;
                        dxi.y += lij * grad.y;
                        dxi.z += lij * grad.z;
                    }
                }
            }
        }

        const float mass_over_rest = (dp.restDensity > 0.f) ? (dp.particleMass / dp.restDensity) : 0.f;
        float3 di = make_float3(mass_over_rest * dxi.x, mass_over_rest * dxi.y, mass_over_rest * dxi.z);

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
    const uint32_t* keysSorted,                 // 新增
    const uint32_t* cellStart,
    const uint32_t* cellEnd,
    sim::DeviceParams dp,
    uint32_t N,
    cudaStream_t s)
{
    const int BS = 256;
    dim3 block(BS), gridDim((N + BS - 1) / BS);

    KDeltaCompute << <gridDim, block, 0, s >> > (delta, pos_pred, lambda, indicesSorted, keysSorted, cellStart, cellEnd, dp, N);
    KApplyDelta   << <gridDim, block, 0, s >> > (pos_pred, delta, indicesSorted, N);
}