#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include "parameters.h"

__device__ inline float len3(const float3& v) { return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z); }

__global__ void StatsBruteKernel(const float4* pos_pred,
    const float4* vel,
    sim::KernelCoeffs kc,
    uint32_t N,
    uint32_t sampleStride,
    uint32_t maxISamples,
    float* gSumNeighbors,
    float* gSumSpeed,
    float* gSumW,
    uint32_t* gSamples)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t tstride = blockDim.x * gridDim.x;

    float locNeighbors = 0.f;
    float locSpeed = 0.f;
    float locSumW = 0.f;
    uint32_t locSamples = 0;
    uint32_t taken = 0;
    for (uint32_t i = tid; i < N && taken < maxISamples; i += tstride) {
        if (sampleStride > 1 && (i % sampleStride) != 0) continue;
        ++taken;

        const float3 pi = make_float3(pos_pred[i].x, pos_pred[i].y, pos_pred[i].z);
        const float3 vi = make_float3(vel[i].x, vel[i].y, vel[i].z);
        locSpeed += len3(vi);

        float neighbors = 0.0f;
        float sumW = 0.0f;

        for (uint32_t j = 0; j < N; ++j) {
            if (j == i) continue;
            const float3 pj = make_float3(pos_pred[j].x, pos_pred[j].y, pos_pred[j].z);
            const float3 d = make_float3(pi.x - pj.x, pi.y - pj.y, pi.z - pj.z);
            const float r2 = d.x * d.x + d.y * d.y + d.z * d.z;
            if (r2 < kc.h2) {
                neighbors += 1.0f;
                const float term = kc.h2 - r2;
                const float w = kc.poly6 * term * term * term;
                sumW += w;
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

extern "C" bool LaunchComputeStatsBruteforce(const float4* pos_pred,
    const float4* vel,
    sim::KernelCoeffs kc,
    float particleMass,
    uint32_t N,
    uint32_t sampleStride,
    uint32_t maxISamples,
    double* outAvgNeighbors,
    double* outAvgSpeed,
    double* outAvgRhoRel,
    double* outAvgRho,
    cudaStream_t s)
{
    if (!pos_pred || !vel || N == 0) {
        if (outAvgNeighbors) *outAvgNeighbors = 0.0;
        if (outAvgSpeed)     *outAvgSpeed = 0.0;
        if (outAvgRhoRel)    *outAvgRhoRel = 0.0;
        if (outAvgRho)       *outAvgRho = 0.0;
        return true;
    }

    float* dSN = nullptr, * dSS = nullptr, * dSW = nullptr;
    uint32_t* dSM = nullptr;
    cudaMalloc(&dSN, sizeof(float));
    cudaMalloc(&dSS, sizeof(float));
    cudaMalloc(&dSW, sizeof(float));
    cudaMalloc(&dSM, sizeof(uint32_t));
    cudaMemsetAsync(dSN, 0, sizeof(float), s);
    cudaMemsetAsync(dSS, 0, sizeof(float), s);
    cudaMemsetAsync(dSW, 0, sizeof(float), s);
    cudaMemsetAsync(dSM, 0, sizeof(uint32_t), s);

    const dim3 bs(128);
    const dim3 gs((maxISamples + bs.x - 1) / bs.x);
    StatsBruteKernel << <gs, bs, 0, s >> > (pos_pred, vel, kc, N, sampleStride, maxISamples, dSN, dSS, dSW, dSM);

    float hSN = 0.f, hSS = 0.f, hSW = 0.f; uint32_t hSM = 0;
    cudaMemcpyAsync(&hSN, dSN, sizeof(float), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(&hSS, dSS, sizeof(float), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(&hSW, dSW, sizeof(float), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(&hSM, dSM, sizeof(uint32_t), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);

    cudaFree(dSN); cudaFree(dSS); cudaFree(dSW); cudaFree(dSM);

    const double samples = (hSM > 0) ? double(hSM) : 1.0;
    const double avgNeighbors = double(hSN) / samples;
    const double avgSpeed = double(hSS) / samples;
    const double avgRhoRel = double(hSW) / samples;
    const double avgRho = avgRhoRel * double(particleMass); 

    if (outAvgNeighbors) *outAvgNeighbors = avgNeighbors;
    if (outAvgSpeed)     *outAvgSpeed = avgSpeed;
    if (outAvgRhoRel)    *outAvgRhoRel = avgRhoRel;
    if (outAvgRho)       *outAvgRho = avgRho;
    return true;
}