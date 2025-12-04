#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace sim {

    // Precomputed smoothing-kernel constants shared across CPU/GPU hot paths.
    struct KernelCoeffs {
        float h;
        float inv_h;
        float h2;
        float poly6;
        float spiky;
        float visc;
    };

    // Axis-aligned simulation container plus derived discretization data.
    struct GridBounds {
        float3 mins;
        float3 maxs;
        float  cellSize;
        int3   dim;
    };

    // PBF-specific toggles kept together so UI->simulation sync is explicit.
    struct PbfTuning {
        int   scorr_enable = 1;
        float scorr_k = 0.003f;
        float scorr_n = 4.0f;
        float scorr_dq_h = 0.3f;
        float wq_min = 1e-12f;
        float scorr_min = -0.25f;

        float grad_r_eps = 1e-6f;
        float lambda_denom_eps = 1e-4f;

        float compliance = 0.0f;
        int   xpbd_enable = 0;

        int   enable_lambda_clamp = 1;
        float lambda_max_abs = 50.0f;
        int   enable_disp_clamp = 1;
        float disp_clamp_max_h = 0.05f;

        int   enable_relax = 1;
        float relax_omega = 0.75f;

        int   xsph_gate_enable = 0;
        int   xsph_n_min = 0;
        int   xsph_n_max = 8;

        int   lambda_warm_start_enable = 0;
        float lambda_warm_start_decay = 0.5f;

        int   semi_implicit_integration_enable = 0;
    };

    // Host-side aggregate of all frequently tuned simulation constants.
    struct SimParams {
        uint32_t numParticles;
        uint32_t maxParticles;
        float3   gravity;
        float    dt;
        float    cfl;
        float    restDensity;
        int      solverIters;
        int      maxNeighbors;
        int      sortEveryN;
        float    boundaryRestitution;
        float    particleMass;
        PbfTuning pbf{};
        float    xsph_c = 0.05f;
        KernelCoeffs kernel{};
        GridBounds  grid{};
        uint32_t ghostParticleCount = 0; // Derived from boundary builders when enabled.
    };

    // POD snapshot copied into constant memory/device-visible buffers.
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
        kc.visc  = 15.0f / (2.0f * pi * h3);
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
