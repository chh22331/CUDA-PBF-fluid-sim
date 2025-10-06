#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include "parameters.h"
#include "stats.h"

__device__ inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
__device__ inline int3 worldToCell(const sim::GridBounds& g, const float3& p) {
    const float3 rel = make_float3((p.x - g.mins.x) / g.cellSize,
        (p.y - g.mins.y) / g.cellSize,
        (p.z - g.mins.z) / g.cellSize);
    int3 c;
    c.x = clampi(int(floorf(rel.x)), 0, g.dim.x - 1);
    c.y = clampi(int(floorf(rel.y)), 0, g.dim.y - 1);
    c.z = clampi(int(floorf(rel.z)), 0, g.dim.z - 1);
    return c;
}
__device__ inline uint32_t cellToLinear(const sim::GridBounds& g, int x, int y, int z) {
    return uint32_t(x + y * g.dim.x + z * g.dim.x * g.dim.y);
}
__device__ inline float length3(const float3& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__global__ void StatsKernel(const float4* pos_pred,
    const float4* vel,
    const uint32_t* indices_sorted,
    const uint32_t* cellStart,
    const uint32_t* cellEnd,
    sim::GridBounds grid,
    sim::KernelCoeffs kc,
    uint32_t N,
    uint32_t numCells,
    uint32_t sampleStride,
    float* gSumNeighbors,
    float* gSumSpeed,
    float* gSumW,
    uint32_t* gSamples)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = blockDim.x * gridDim.x;

    float locNeighbors = 0.f;
    float locSpeed = 0.f;
    float locSumW = 0.f;
    uint32_t locSamples = 0;

    for (uint32_t i = tid; i < N; i += stride) {
        if (sampleStride > 1 && (i % sampleStride) != 0) continue;
        const float3 pi = make_float3(pos_pred[i].x, pos_pred[i].y, pos_pred[i].z);
        const float3 vi = make_float3(vel[i].x, vel[i].y, vel[i].z);
        locSpeed += length3(vi);

        const int3 c = worldToCell(grid, pi);
        float neighbors = 0.0f;
        float sumW = 0.0f;

        for (int dz = -1; dz <= 1; ++dz) {
            int z = c.z + dz; if (z < 0 || z >= grid.dim.z) continue;
            for (int dy = -1; dy <= 1; ++dy) {
                int y = c.y + dy; if (y < 0 || y >= grid.dim.y) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    int x = c.x + dx; if (x < 0 || x >= grid.dim.x) continue;
                    uint32_t cellId = cellToLinear(grid, x, y, z);
                    if (cellId >= numCells) continue;
                    const uint32_t begin = cellStart[cellId];
                    const uint32_t end = cellEnd[cellId];
                    if (begin == 0xFFFFFFFFu || begin >= end) continue; // 补充哨兵检查

                    for (uint32_t k = begin; k < end; ++k) {
                        const uint32_t j = indices_sorted[k];
                        if (j == i) continue;
                        const float3 pj = make_float3(pos_pred[j].x, pos_pred[j].y, pos_pred[j].z);
                        const float3 d = make_float3(pi.x - pj.x, pi.y - pj.y, pi.z - pj.z);
                        const float r2 = d.x * d.x + d.y * d.y + d.z * d.z;
                        if (r2 < kc.h2) {
                            neighbors += 1.0f;
                            const float term = (kc.h2 - r2);
                            const float w = kc.poly6 * term * term * term; // poly6
                            sumW += w;
                        }
                    }
                }
            }
        }

        locNeighbors += neighbors;
        locSumW += sumW;
        ++locSamples;
    }

    if (locSamples > 0) {
        atomicAdd(gSumNeighbors, locNeighbors);
        atomicAdd(gSumSpeed, locSpeed);
        atomicAdd(gSumW, locSumW);
        atomicAdd(gSamples, locSamples);
    }
}

extern "C" bool LaunchComputeStats(const float4* pos_pred,
    const float4* vel,
    const uint32_t* indices_sorted,
    const uint32_t* cellStart,
    const uint32_t* cellEnd,
    sim::GridBounds grid,
    sim::KernelCoeffs kc,
    float particleMass,
    uint32_t N,
    uint32_t numCells,
    uint32_t sampleStride,
    double* outAvgNeighbors,
    double* outAvgSpeed,
    double* outAvgRhoRel,
    double* outAvgRho,
    cudaStream_t s)
{
    if (!pos_pred || !vel || !indices_sorted || !cellStart || !cellEnd || N == 0) {
        if (outAvgNeighbors) *outAvgNeighbors = 0.0;
        if (outAvgSpeed)     *outAvgSpeed = 0.0;
        if (outAvgRhoRel)    *outAvgRhoRel = 0.0;
        if (outAvgRho)       *outAvgRho = 0.0;
        return true;
    }

    float* dSumNeighbors = nullptr, * dSumSpeed = nullptr, * dSumW = nullptr;
    uint32_t* dSamples = nullptr;
    cudaMalloc(&dSumNeighbors, sizeof(float));
    cudaMalloc(&dSumSpeed, sizeof(float));
    cudaMalloc(&dSumW, sizeof(float));
    cudaMalloc(&dSamples, sizeof(uint32_t));
    cudaMemsetAsync(dSumNeighbors, 0, sizeof(float), s);
    cudaMemsetAsync(dSumSpeed, 0, sizeof(float), s);
    cudaMemsetAsync(dSumW, 0, sizeof(float), s);
    cudaMemsetAsync(dSamples, 0, sizeof(uint32_t), s);

    const dim3 bs(256);
    const dim3 gs((N + bs.x - 1) / bs.x);
    StatsKernel<<<gs, bs, 0, s>>>(pos_pred, vel, indices_sorted, cellStart, cellEnd,
        grid, kc, N, numCells, sampleStride, dSumNeighbors, dSumSpeed, dSumW, dSamples);

    float hSumNeighbors = 0.f, hSumSpeed = 0.f, hSumW = 0.f;
    uint32_t hSamples = 0;
    cudaMemcpyAsync(&hSumNeighbors, dSumNeighbors, sizeof(float), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(&hSumSpeed, dSumSpeed, sizeof(float), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(&hSumW, dSumW, sizeof(float), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(&hSamples, dSamples, sizeof(uint32_t), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);

    cudaFree(dSumNeighbors);
    cudaFree(dSumSpeed);
    cudaFree(dSumW);
    cudaFree(dSamples);

    const double samples = (hSamples > 0) ? double(hSamples) : 1.0;
    const double avgNeighbors = double(hSumNeighbors) / samples;
    const double avgSpeed = double(hSumSpeed) / samples;
    const double avgRhoRel = double(hSumW) / samples;
    const double avgRho = avgRhoRel * double(particleMass); // 修正：ρ = m * sumW

    if (outAvgNeighbors) *outAvgNeighbors = avgNeighbors;
    if (outAvgSpeed)     *outAvgSpeed = avgSpeed;
    if (outAvgRhoRel)    *outAvgRhoRel = avgRhoRel;
    if (outAvgRho)       *outAvgRho = avgRho;

    return true;
}