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

    // 调参与稳健性
    struct PbfTuning {
        // —— tensile instability 修正 s_corr = -k * (W(r)/W(dq))^n
        int   scorr_enable = 1;
        float scorr_k = 0.02f;       // 原 0.003f -> 0.02f，显著减轻聚团
        float scorr_n = 4.0f;
        float scorr_dq_h = 0.3f;
        float wq_min = 1e-12f;

        // 梯度与 λ 分母正则（略增稳健）
        float grad_r_eps = 1e-6f;
        float lambda_denom_eps = 1e-4f; // 原 1e-5f -> 1e-4f，缓解 λ 过大

        // λ 与位移钳制（略收紧位移）
        int   enable_lambda_clamp = 1;
        float lambda_max_abs = 50.0f;
        int   enable_disp_clamp = 1;
        float disp_clamp_max_h = 0.05f; // 原 0.1f -> 0.05f

        // —— XSPH 稀疏区门控（新增）
        int   xsph_gate_enable = 1;  // 1=按邻居数门控
        int   xsph_n_min = 8;        // n <= 8 基本不施加 XSPH
        int   xsph_n_max = 28;       // n >= 28 完全施加 XSPH
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

        float    xsph_c = 0.0f; // 控制台下发

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
        dp.pbf = sp.pbf;      // 包含新门控参数
        dp.xsph_c = sp.xsph_c;
        return dp;
    }

} // namespace sim