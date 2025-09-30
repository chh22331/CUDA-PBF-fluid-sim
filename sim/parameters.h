#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace sim {

    // Common kernel coefficients (SPH/PBF)
    struct KernelCoeffs {
        float h;
        float inv_h;
        float h2;
        float poly6;  // 315/(64*pi*h^9)
        float spiky;  // 15/(pi*h^6)
        float visc;   // 15/(2*pi*h^3)
    };

    struct GridBounds {
        float3 mins;
        float3 maxs;
        float  cellSize; // usually = h
        int3   dim;      // ceil((max-min)/cellSize)
    };

    struct SimParams {
        uint32_t numParticles = 0;
        float    dt = 0.004f;
        float    cfl = 0.45f;
        float3   gravity = make_float3(0.0f, -9.8f, 0.0f);
        float    restDensity = 1000.0f;
        int      solverIters = 4;
        int      maxNeighbors = 64;
        bool     useMixedPrecision = true;
        int      sortEveryN = 1;
        KernelCoeffs kernel;
        GridBounds  grid;
    };

} // namespace sim