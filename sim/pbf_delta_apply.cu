#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "parameters.h"
#include "cuda_vec_math.cuh"

namespace {

    __device__ __forceinline__
    bool inBounds(int3 c, int3 dim) {
        return (c.x >= 0 && c.x < dim.x && c.y >= 0 && c.y < dim.y && c.z >= 0 && c.z < dim.z);
    }
    __device__ __forceinline__
    uint32_t lid(int3 c, int3 dim) { return (uint32_t)((c.z * dim.y + c.y) * dim.x + c.x); }

    __global__ void KDelta(float4* pos_pred, float4* delta, const float* lambda,
        const uint32_t* indicesSorted,
        const uint32_t* cellStart, const uint32_t* cellEnd,
        sim::GridBounds grid, sim::KernelCoeffs kc,
        int /*maxNeighbors*/, uint32_t N) {
        uint32_t sortedIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (sortedIdx >= N) return;

        uint32_t i = indicesSorted[sortedIdx];
        float3 pi = to_float3(pos_pred[i]);

        float3 rel = make_float3((pi.x - grid.mins.x) / grid.cellSize,
            (pi.y - grid.mins.y) / grid.cellSize,
            (pi.z - grid.mins.z) / grid.cellSize);
        int3 ci = make_int3(floorf(rel.x), floorf(rel.y), floorf(rel.z));

        float3 dxi = make_float3(0.f, 0.f, 0.f);

        for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    int3 cc = make_int3(ci.x + dx, ci.y + dy, ci.z + dz);
                    if (!inBounds(cc, grid.dim)) continue;
                    uint32_t cidx = lid(cc, grid.dim);
                    uint32_t beg = cellStart[cidx];
                    uint32_t end = cellEnd[cidx];
                    if (beg == 0xFFFFFFFFu || beg >= end) continue;

                    for (uint32_t k = beg; k < end; ++k) {
                        uint32_t j = indicesSorted[k];
                        if (j == i) continue;
                        float3 pj = to_float3(pos_pred[j]);
                        float3 rij = pi - pj;
                        float r2 = rij.x * rij.x + rij.y * rij.y + rij.z * rij.z;
                        if (r2 > kc.h2) continue;
                        float r = sqrtf(r2);
                        float t = kc.h - r;
                        float scorr = 0.0f; // 可选人工压力项
                        float grad = (r > 1e-6f) ? (kc.spiky * t * t / r) : 0.f;
                        float3 n = make_float3(grad * rij.x, grad * rij.y, grad * rij.z);
                        float w = 1.0f; // 质量相同
                        float lij = lambda[i] + lambda[j] + scorr;
                        dxi.x += w * lij * n.x;
                        dxi.y += w * lij * n.y;
                        dxi.z += w * lij * n.z;
                    }
                }

        // 位置更新
        float3 pi_new = make_float3(pi.x + dxi.x, pi.y + dxi.y, pi.z + dxi.z);
        pos_pred[i] = make_float4(pi_new.x, pi_new.y, pi_new.z, 1.0f);
        delta[i] = make_float4(dxi.x, dxi.y, dxi.z, 0.0f);
    }

} // anon

extern "C" void LaunchDeltaApply(float4* pos_pred, float4* delta, const float* lambda, const uint32_t* indicesSorted,
    const uint32_t* cellStart, const uint32_t* cellEnd,
    sim::GridBounds grid, sim::KernelCoeffs kc, int maxNeighbors, uint32_t N, cudaStream_t s) {
    const int BS = 256;
    dim3 block(BS), gridDim((N + BS - 1) / BS);
    KDelta<<<gridDim, block, 0, s>>>(pos_pred, delta, lambda, indicesSorted, cellStart, cellEnd, grid, kc, maxNeighbors, N);
}