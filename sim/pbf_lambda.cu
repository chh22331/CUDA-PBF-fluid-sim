#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "parameters.h"
#include "cuda_vec_math.cuh"

namespace {

    __device__ __forceinline__
    bool cellInBounds(int3 c, int3 dim) {
        return (c.x >= 0 && c.x < dim.x && c.y >= 0 && c.y < dim.y && c.z >= 0 && c.z < dim.z);
    }

    __device__ __forceinline__
    uint32_t linIdx(int3 c, int3 dim) {
        return (uint32_t)((c.z * dim.y + c.y) * dim.x + c.x);
    }

    __global__ void KLambda(float* lambda, const float4* pos_pred,
                            const uint32_t* indicesSorted,
                            const uint32_t* cellStart, const uint32_t* cellEnd,
                            sim::GridBounds grid, sim::KernelCoeffs kc,
                            float restDensity, int /*maxNeighbors*/, uint32_t N) {
        uint32_t sortedIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (sortedIdx >= N) return;

        uint32_t i = indicesSorted[sortedIdx];
        float3 pi = to_float3(pos_pred[i]);

        // particle i cell
        float3 rel = make_float3((pi.x - grid.mins.x) / grid.cellSize,
                                 (pi.y - grid.mins.y) / grid.cellSize,
                                 (pi.z - grid.mins.z) / grid.cellSize);
        int3 ci = make_int3(floorf(rel.x), floorf(rel.y), floorf(rel.z));

        float density = 0.f;
        float3 gradSum = make_float3(0.f, 0.f, 0.f);
        float sumGrad2 = 0.f;

        // 27-neighborhood
        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            int3 cc = make_int3(ci.x + dx, ci.y + dy, ci.z + dz);
            if (!cellInBounds(cc, grid.dim)) continue;
            uint32_t cidx = linIdx(cc, grid.dim);
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

                // poly6 density kernel: (h^2 - r^2)^3
                float hr2 = kc.h2 - r2;
                float w = kc.poly6 * hr2 * hr2 * hr2;
                density += w;

                // spiky gradient: -45/(pi*h^6) * (h - r)^2 / r * (rij)
                float r = sqrtf(r2);
                float t = kc.h - r;
                // 预计算 -45/(pi*h^6) = -3 * kc.spiky （若 kc.spiky=15/(pi*h^6)）
                // 为避免依赖 kc.spiky 的取值，这里直接使用常数表达式：
                const float inv_h6 = 1.0f / (kc.h2 * kc.h2 * kc.h2);
                const float coeff = (r > 1e-6f) ? (-45.0f * inv_h6 * (t * t) / r) : 0.0f;
                float3 grad = make_float3(coeff * rij.x, coeff * rij.y, coeff * rij.z);

                gradSum.x += grad.x; gradSum.y += grad.y; gradSum.z += grad.z;
                sumGrad2 += grad.x * grad.x + grad.y * grad.y + grad.z * grad.z;
            }
        }

        // 自项：|sum_j grad C_i|^2
        sumGrad2 += (gradSum.x * gradSum.x + gradSum.y * gradSum.y + gradSum.z * gradSum.z);

        float C = density / restDensity - 1.f;
        float denom = sumGrad2 + 1e-6f;
        lambda[i] = -C / denom;
    }

} // anon

extern "C" void LaunchLambda(float* lambda, const float4* pos_pred, const uint32_t* indicesSorted,
                             const uint32_t* cellStart, const uint32_t* cellEnd,
                             sim::GridBounds grid, sim::KernelCoeffs kc,
                             float restDensity, int maxNeighbors, uint32_t N, cudaStream_t s) {
    const int BS = 256;
    dim3 block(BS), gridDim((N + BS - 1) / BS);
    KLambda<<<gridDim, block, 0, s>>>(lambda, pos_pred, indicesSorted, cellStart, cellEnd, grid, kc, restDensity, maxNeighbors, N);
}