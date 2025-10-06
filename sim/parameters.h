#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace sim {

    struct KernelCoeffs {
        float h;
        float inv_h;
        float h2;
        float poly6;
        float spiky;
        float visc;
    };

    struct GridBounds {
        float3 mins;
        float3 maxs;
        float  cellSize;
        int3   dim;
    };

    // 集中化的 PBF 调参项（不再包含 xsph 系数）
    struct PbfTuning {
        float scorr_k = 0.003f;
        float scorr_n = 4.0f;
        float scorr_dq_h = 0.3f;
        float grad_r_eps = 1e-6f;
        float lambda_denom_eps = 1e-6f;
        float wq_min = 1e-12f;
    };

    struct SimParams {
        uint32_t numParticles;
        uint32_t maxParticles;

        float3   gravity;
        float    dt;
        float    cfl;
        float    restDensity;
        int      solverIters;
        int      maxNeighbors;
        bool     useMixedPrecision;
        int      sortEveryN;
        float    boundaryRestitution;

        float    particleMass;
        PbfTuning pbf{};

        // 新增：XSPH 系数从 console 下发
        float    xsph_c = 0.0f;

        KernelCoeffs kernel{};
        GridBounds  grid{};
    };

    struct DeviceParams {
        KernelCoeffs kernel;
        GridBounds   grid;
        float3       gravity;
        float        restDensity;
        float        dt;
        float        inv_dt;
        float        boundaryRestitution;
        int          maxNeighbors;
        float        particleMass;
        PbfTuning    pbf;

        // 新增：XSPH 系数（与 SimParams 对齐）
        float        xsph_c;
    };

    inline KernelCoeffs MakeKernelCoeffs(float h) {
        const float pi = 3.14159265358979323846f;
        KernelCoeffs kc{};
        kc.h = h;
        kc.inv_h = 1.0f / h;
        kc.h2 = h * h;
        const float h3 = h * kc.h2;
        const float h6 = h3 * h3;
        const float h9 = h6 * h3;
        kc.poly6 = 315.0f / (64.0f * pi * h9);
        kc.spiky = 15.0f / (pi * h6);
        kc.visc = 15.0f / (2.0f * pi * h3);
        return kc;
    }

    inline DeviceParams MakeDeviceParams(const SimParams& sp) {
        DeviceParams dp{};
        dp.kernel = sp.kernel;
        dp.grid = sp.grid;
        dp.gravity = sp.gravity;
        dp.restDensity = sp.restDensity;
        dp.dt = sp.dt;
        dp.inv_dt = (sp.dt > 0.f) ? (1.0f / sp.dt) : 0.0f;
        dp.boundaryRestitution = sp.boundaryRestitution;
        dp.maxNeighbors = sp.maxNeighbors;
        dp.particleMass = sp.particleMass;
        dp.pbf = sp.pbf;
        dp.xsph_c = sp.xsph_c;
        return dp;
    }

} // namespace sim