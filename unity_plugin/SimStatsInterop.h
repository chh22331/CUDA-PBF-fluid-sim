#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "../sim/stats.h"

struct SimStatsInterop {
    uint32_t N;
    double   avgNeighbors;
    double   avgSpeed;
    double   avgRho;
    double   avgRhoRel;
    uint32_t reserved0;
    uint32_t reserved1;
};

inline void ConvertToInterop(const sim::SimStats& in, SimStatsInterop& out) {
    out.N = in.N;
    out.avgNeighbors = in.avgNeighbors;
    out.avgSpeed = in.avgSpeed;
    out.avgRho = in.avgRho;
    out.avgRhoRel = in.avgRhoRel;
    out.reserved0 = 0;
    out.reserved1 = 0;
}