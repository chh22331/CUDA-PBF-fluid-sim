#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace sim {

    // ========== 新增：与 console 同步的数值类型枚举（基础结构 M1）==========
    enum class NumericType : uint8_t {
        FP32 = 0,
        FP16 = 1,
        FP16_Packed = 2,
        Quantized16 = 3,
        InvalidSentinel = 255
    };

    // ========== 新增：精度配置（运行期由 BuildSimParams 填充）==========
    struct SimPrecision {
        // 存储（位置 /速度 /预测 / λ / 密度 / 辅助 / 渲染提交转换）
        NumericType positionStore      = NumericType::FP32;
        NumericType velocityStore      = NumericType::FP32;
        NumericType predictedPosStore  = NumericType::FP32;
        NumericType lambdaStore        = NumericType::FP32; // 可半精镜像（__half 标量）
        NumericType densityStore       = NumericType::FP32; // 可半精镜像（__half 标量）
        NumericType auxStore           = NumericType::FP32; // 可半精镜像（__half 标量）
        NumericType renderTransfer     = NumericType::FP32;

        // 计算
        NumericType coreCompute        = NumericType::FP32;
        bool        forceFp32Accumulate = true;
        bool        enableHalfIntrinsics = false;

        // 分阶段覆盖
        bool        useStageOverrides  = false;
        NumericType emissionCompute    = NumericType::FP32;
        NumericType gridBuildCompute   = NumericType::FP32;
        NumericType neighborCompute    = NumericType::FP32;
        NumericType densityCompute     = NumericType::FP32;
        NumericType lambdaCompute      = NumericType::FP32;
        NumericType integrateCompute   = NumericType::FP32;
        NumericType velocityCompute    = NumericType::FP32;
        NumericType boundaryCompute    = NumericType::FP32;
        NumericType xsphCompute        = NumericType::FP32;

        uint32_t    fp16StageMask      = 0;

        // 自适应（预留）
        bool        adaptivePrecision = false;
        float       densityErrorTolerance = 0.01f;
        float       lambdaVarianceTolerance = 0.05f;
        int         adaptCheckEveryN = 30;

        NumericType _adaptive_pos_prev = NumericType::InvalidSentinel;
        NumericType _adaptive_vel_prev = NumericType::InvalidSentinel;
        NumericType _adaptive_pos_pred_prev = NumericType::InvalidSentinel;

        bool nativeHalfActive = false; // 新增：原生 half4 主存储激活
    };

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
        float scorr_k = 0.003f;
        float scorr_n = 4.0f;
        float scorr_dq_h = 0.3f;
        float wq_min = 1e-12f;
        float scorr_min = -0.25f;

        float grad_r_eps = 1e-6f;
        float lambda_denom_eps = 1e-4f;

        // ====== XPBD 合规参数 ======
        float compliance = 0.0f;        // α = compliance / dt^2; 当 xpbd_enable=0 或 compliance=0 -> 退化为 PBF
        int   xpbd_enable = 0;          // 新增：是否启用 XPBD 修正

        int   enable_lambda_clamp = 1;
        float lambda_max_abs = 50.0f;
        int   enable_disp_clamp = 1;
        float disp_clamp_max_h = 0.05f;

        int   enable_relax = 1;
        float relax_omega = 0.75f;

        int   xsph_gate_enable = 0;
        int   xsph_n_min = 0;
        int   xsph_n_max = 8;

        // ====== λ Warm-Start 控制 ======
        int   lambda_warm_start_enable = 0;
        float lambda_warm_start_decay = 0.5f;  // 0~1

        // ====== 半隐式积分开关（供积分核使用） ======
        int   semi_implicit_integration_enable = 0;
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
        float    xsph_c = 0.05f;

        KernelCoeffs kernel{};
        GridBounds  grid{};

        // ========== 新增：映射后的精度配置（供设备侧使用）==========
        SimPrecision precision{};

        // ======== 新增：幽灵边界粒子计数（静态，设备侧用于邻域贡献，不参与积分） ========
        uint32_t ghostParticleCount = 0; // 由 Simulator 在生成后填充

        // ===== 哈希压缩网格新增参数 =====
        bool     useHashedGrid = true;    // perf.use_hashed_grid
        int      compactRebuildEveryN = 32; // perf.sort_compact_every_n (>=1)
        bool     compactBinarySearch = true; // perf.compact_binary_search
        bool     logGridCompactStats = false; // perf.log_grid_compact_stats
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
        // 可选：以后扩展加入 precision 快速判定
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