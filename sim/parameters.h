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

    struct PbfTuning {
        int   scorr_enable = 1;
        float scorr_k = 0.012f;      // 原 0.003f -> 略增强排斥，抑制穿插/回弹
        float scorr_n = 4.0f;
        float scorr_dq_h = 0.3f;
        float wq_min = 1e-12f;
        float scorr_min = -0.25f;    // 放宽负向上限，近接时更有力

        float grad_r_eps = 1e-6f;
        float lambda_denom_eps = 1e-4f;

        float compliance = 0.0f;     // 关闭 XPBD 软度，先确保可静止

        int   enable_lambda_clamp = 1;
        float lambda_max_abs = 50.0f;
        int   enable_disp_clamp = 1;
        float disp_clamp_max_h = 0.10f; // 放宽单步位移夹取，提高收敛

        int   enable_relax = 1;
        float relax_omega = 0.75f;   // 略加强阻尼，防迭代过冲

        // 建议先关门控，或降低门槛以保证靠近边界仍有阻尼
        int   xsph_gate_enable = 0;  // 先禁用门控验证
        int   xsph_n_min = 0;
        int   xsph_n_max = 8;
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

        float    xsph_c = 0.05f;     // 默认开启一定阻尼

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