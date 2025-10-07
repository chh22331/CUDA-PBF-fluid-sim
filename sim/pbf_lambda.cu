#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>
#include "parameters.h"
#include "cuda_vec_math.cuh"
#include "cuda_grid_utils.cuh"

namespace {

    __global__ void KLambda(float* lambda, const float4* pos_pred,
        const uint32_t* indicesSorted,
        const uint32_t* keysSorted,                 // 新增
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

        // 用排序后的 key 还原单元坐标（与 cellStart/cellEnd 一致）
        const uint32_t key = keysSorted[sortedIdx];
        int3 ci;
        ci.x = int(key % uint32_t(grid.dim.x));
        uint32_t key_div_x = key / uint32_t(grid.dim.x);
        ci.y = int(key_div_x % uint32_t(grid.dim.y));
        ci.z = int(key_div_x / uint32_t(grid.dim.y));

        float density = 0.f;
        float3 gradSum = make_float3(0.f, 0.f, 0.f);
        float sumGrad2 = 0.f;

        // 自项密度
        {
            const float hr2 = kc.h2;
            const float w0 = kc.poly6 * hr2 * hr2 * hr2;
            density += particleMass * w0;
        }

        // 自适应邻域层数：覆盖半径 h
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

                        float hr2 = kc.h2 - r2;
                        float w = kc.poly6 * hr2 * hr2 * hr2;
                        density += particleMass * w;

                        float r = sqrtf(r2);
                        float t = kc.h - r;
                        const float coeff = (r > pbf.grad_r_eps) ? (-3.0f * kc.spiky * (t * t) / r) : 0.0f;
                        float3 grad = make_float3(coeff * rij.x, coeff * rij.y, coeff * rij.z);

                        gradSum.x += grad.x; gradSum.y += grad.y; gradSum.z += grad.z;
                        sumGrad2 += grad.x * grad.x + grad.y * grad.y + grad.z * grad.z;
                    }
                }
            }
        }

        const float gradSum2 = gradSum.x * gradSum.x + gradSum.y * gradSum.y + gradSum.z * gradSum.z;
        const float scale = (restDensity > 0.f) ? (particleMass / restDensity) : 0.f;
        float denom = (scale * scale) * (sumGrad2 + gradSum2) + pbf.lambda_denom_eps;
        denom = fmaxf(denom, pbf.lambda_denom_eps);

        const float C = density / restDensity - 1.f;
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
    const uint32_t* keysSorted,                     // 新增
    const uint32_t* cellStart, const uint32_t* cellEnd,
    sim::DeviceParams dp,
    uint32_t N, cudaStream_t s) {
    const int BS = 256;
    dim3 block(BS), gridDim((N + BS - 1) / BS);
    KLambda << <gridDim, block, 0, s >> > (lambda, pos_pred, indicesSorted, keysSorted, cellStart, cellEnd, dp, N);
}