#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "parameters.h"

namespace sim {
    struct SimStats {
        uint32_t N = 0;
        double   avgNeighbors = 0;
        double   avgSpeed = 0;
        double   avgRhoRel = 0;  // = avg(sum_j W_ij)
        double   avgRho = 0;     // = particleMass * avgRhoRel
    };
}

// Extern kernels placed outside namespace to avoid NVCC duplicated name mangling ambiguity
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
    cudaStream_t s);

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
    cudaStream_t s);
