#pragma once
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include "sim/parameters.h" // sim::SimParams / DeviceParams / PbfTuning / KernelCoeffs / GridBounds

// 互操作：PbfTuning
struct PbfTuningInterop {
    int   scorr_enable;
    float scorr_k;
    float scorr_n;
    float scorr_dq_h;
    float wq_min;
    float scorr_min;

    float grad_r_eps;
    float lambda_denom_eps;

    float compliance;
    int   xpbd_enable;

    int   enable_lambda_clamp;
    float lambda_max_abs;
    int   enable_disp_clamp;
    float disp_clamp_max_h;

    int   enable_relax;
    float relax_omega;

    int   xsph_gate_enable;
    int   xsph_n_min;
    int   xsph_n_max;

    int   lambda_warm_start_enable;
    float lambda_warm_start_decay;

    int   semi_implicit_integration_enable;
};

// 互操作：Kernel / Grid
struct KernelCoeffsInterop {
    float h;
    float inv_h;
    float h2;
    float poly6;
    float spiky;
    float visc;
};
struct GridBoundsInterop {
    float3 mins;
    float3 maxs;
    float  cellSize;
    int3   dim; // 由 native 侧 buildGrid 计算，interop 可忽略填写
};

// 互操作：SimParams（严格覆盖 sim::SimParams 字段顺序与类型）
struct SimParamsInterop {
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
    PbfTuningInterop pbf;
    float    xsph_c;
    KernelCoeffsInterop kernel;
    GridBoundsInterop   grid;
    uint32_t ghostParticleCount;
    float    maxSpeedClamp; // <0 禁用
    uint32_t version;       // 互操作版本，当前=1
};

// 互操作：DeviceParams（如需导出）
struct DeviceParamsInterop {
    KernelCoeffsInterop kernel;
    GridBoundsInterop   grid;
    float3       gravity;
    float        restDensity;
    float        dt;
    float        inv_dt;
    float        boundaryRestitution;
    int          maxNeighbors;
    float        particleMass;
    PbfTuningInterop pbf;
    float        xsph_c;
};

// --- 转换 ---
inline void ConvertToNative(const PbfTuningInterop& in, sim::PbfTuning& out) {
    out.scorr_enable = in.scorr_enable;
    out.scorr_k = in.scorr_k;
    out.scorr_n = in.scorr_n;
    out.scorr_dq_h = in.scorr_dq_h;
    out.wq_min = in.wq_min;
    out.scorr_min = in.scorr_min;

    out.grad_r_eps = in.grad_r_eps;
    out.lambda_denom_eps = in.lambda_denom_eps;

    out.compliance = in.compliance;
    out.xpbd_enable = in.xpbd_enable;

    out.enable_lambda_clamp = in.enable_lambda_clamp;
    out.lambda_max_abs = in.lambda_max_abs;
    out.enable_disp_clamp = in.enable_disp_clamp;
    out.disp_clamp_max_h = in.disp_clamp_max_h;

    out.enable_relax = in.enable_relax;
    out.relax_omega = in.relax_omega;

    out.xsph_gate_enable = in.xsph_gate_enable;
    out.xsph_n_min = in.xsph_n_min;
    out.xsph_n_max = in.xsph_n_max;

    out.lambda_warm_start_enable = in.lambda_warm_start_enable;
    out.lambda_warm_start_decay = in.lambda_warm_start_decay;

    out.semi_implicit_integration_enable = in.semi_implicit_integration_enable;
}
inline void ConvertToInterop(const sim::PbfTuning& in, PbfTuningInterop& out) {
    out.scorr_enable = in.scorr_enable;
    out.scorr_k = in.scorr_k;
    out.scorr_n = in.scorr_n;
    out.scorr_dq_h = in.scorr_dq_h;
    out.wq_min = in.wq_min;
    out.scorr_min = in.scorr_min;

    out.grad_r_eps = in.grad_r_eps;
    out.lambda_denom_eps = in.lambda_denom_eps;

    out.compliance = in.compliance;
    out.xpbd_enable = in.xpbd_enable;

    out.enable_lambda_clamp = in.enable_lambda_clamp;
    out.lambda_max_abs = in.lambda_max_abs;
    out.enable_disp_clamp = in.enable_disp_clamp;
    out.disp_clamp_max_h = in.disp_clamp_max_h;

    out.enable_relax = in.enable_relax;
    out.relax_omega = in.relax_omega;

    out.xsph_gate_enable = in.xsph_gate_enable;
    out.xsph_n_min = in.xsph_n_min;
    out.xsph_n_max = in.xsph_n_max;

    out.lambda_warm_start_enable = in.lambda_warm_start_enable;
    out.lambda_warm_start_decay = in.lambda_warm_start_decay;

    out.semi_implicit_integration_enable = in.semi_implicit_integration_enable;
}

inline void ConvertToNative(const KernelCoeffsInterop& in, sim::KernelCoeffs& out) {
    out.h = in.h;
    out.inv_h = in.inv_h;
    out.h2 = in.h2;
    out.poly6 = in.poly6;
    out.spiky = in.spiky;
    out.visc = in.visc;
}
inline void ConvertToInterop(const sim::KernelCoeffs& in, KernelCoeffsInterop& out) {
    out.h = in.h;
    out.inv_h = in.inv_h;
    out.h2 = in.h2;
    out.poly6 = in.poly6;
    out.spiky = in.spiky;
    out.visc = in.visc;
}

inline void ConvertToNative(const GridBoundsInterop& in, sim::GridBounds& out) {
    out.mins = in.mins;
    out.maxs = in.maxs;
    out.cellSize = in.cellSize;
    out.dim = in.dim; // 将在 buildGrid/updateGrid 时按需刷新
}
inline void ConvertToInterop(const sim::GridBounds& in, GridBoundsInterop& out) {
    out.mins = in.mins;
    out.maxs = in.maxs;
    out.cellSize = in.cellSize;
    out.dim = in.dim;
}

inline void ConvertToNative(const SimParamsInterop& in, sim::SimParams& out) {
    out.numParticles = in.numParticles;
    out.maxParticles = in.maxParticles;
    out.gravity = in.gravity;
    out.dt = in.dt;
    out.cfl = in.cfl;
    out.restDensity = in.restDensity;
    out.solverIters = in.solverIters;
    out.maxNeighbors = in.maxNeighbors;
    out.sortEveryN = in.sortEveryN;
    out.boundaryRestitution = in.boundaryRestitution;
    out.particleMass = in.particleMass;
    ConvertToNative(in.pbf, out.pbf);
    out.xsph_c = in.xsph_c;
    // kernel：若只给了 h，自动计算其它系数
    sim::KernelCoeffs kc{};
    ConvertToNative(in.kernel, kc);
    if (kc.h > 0.f && (kc.poly6 == 0.f || kc.spiky == 0.f || kc.visc == 0.f)) {
        kc = sim::MakeKernelCoeffs(kc.h);
    }
    out.kernel = kc;
    ConvertToNative(in.grid, out.grid);
    out.ghostParticleCount = in.ghostParticleCount;
    out.maxSpeedClamp = in.maxSpeedClamp;
}

inline void ConvertToInterop(const sim::SimParams& in, SimParamsInterop& out) {
    std::memset(&out, 0, sizeof(out));
    out.numParticles = in.numParticles;
    out.maxParticles = in.maxParticles;
    out.gravity = in.gravity;
    out.dt = in.dt;
    out.cfl = in.cfl;
    out.restDensity = in.restDensity;
    out.solverIters = in.solverIters;
    out.maxNeighbors = in.maxNeighbors;
    out.sortEveryN = in.sortEveryN;
    out.boundaryRestitution = in.boundaryRestitution;
    out.particleMass = in.particleMass;
    ConvertToInterop(in.pbf, out.pbf);
    out.xsph_c = in.xsph_c;
    ConvertToInterop(in.kernel, out.kernel);
    ConvertToInterop(in.grid, out.grid);
    out.ghostParticleCount = in.ghostParticleCount;
    out.maxSpeedClamp = in.maxSpeedClamp;
    out.version = 1;
}

inline void ConvertToInterop(const sim::DeviceParams& in, DeviceParamsInterop& out) {
    ConvertToInterop(in.kernel, out.kernel);
    ConvertToInterop(in.grid, out.grid);
    out.gravity = in.gravity;
    out.restDensity = in.restDensity;
    out.dt = in.dt;
    out.inv_dt = in.inv_dt;
    out.boundaryRestitution = in.boundaryRestitution;
    out.maxNeighbors = in.maxNeighbors;
    out.particleMass = in.particleMass;
    ConvertToInterop(in.pbf, out.pbf);
    out.xsph_c = in.xsph_c;
}

// 轻量断言（可按需注释）
static_assert(sizeof(SimParamsInterop) % 4 == 0, "SimParamsInterop alignment");