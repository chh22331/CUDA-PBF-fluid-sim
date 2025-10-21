#pragma once
#include <cstdint>
#include <string>
#include <cuda_runtime.h> // for float3/make_float3, int3/make_int3
#include "../../sim/parameters.h"
#include "../../engine/gfx/renderer.h"
#include "../../sim/emit_params.h"

namespace console {
    // ================== 数值精度枚举与配置（增量：不影响现有 FP32 逻辑） ================== //
    // 存储 / 计算可选类型；后续可在构建 SimParams 时映射到具体 kernel/缓冲分配
    enum class NumericType : uint8_t {
        FP32 = 0,        // 默认：float
        FP16 = 1,        // __half (标量或 SoA)
        FP16_Packed = 2, // 打包布局（如 __half4 / AoSoA），提高带宽利用
        Quantized16 = 3  // 预留：定点/自定义 16-bit（暂未实现）
    };

    // 计算阶段标识（位掩码，可用于统一快速切换），仅作为运行时配置描述，不影响现有逻辑
    enum ComputeStageBits : uint32_t {
        Stage_Emission = 1u << 0,
        Stage_GridBuild = 1u << 1,
        Stage_NeighborGather = 1u << 2,
        Stage_Density = 1u << 3,
        Stage_LambdaSolve = 1u << 4,
        Stage_Integration = 1u << 5,
        Stage_VelocityUpdate = 1u << 6,
        Stage_Boundary = 1u << 7,
        Stage_XSPH = 1u << 8,
        Stage_AdaptiveH = 1u << 9,
        Stage_Diagnostics = 1u << 10
    };

    // 统一数值精度配置（作为 Simulation 的子成员），仅定义开关，无副作用
    struct PrecisionConfig {
        // ---------------- 存储精度（粒子主数据与辅助缓冲） ----------------
        NumericType positionStore = NumericType::FP32;
        NumericType velocityStore = NumericType::FP32;
        NumericType predictedPosStore = NumericType::FP32; // PBF 预测位置（若使用）
        NumericType lambdaStore = NumericType::FP32; // PBF λ（约束求解敏感，初期保持 FP32）
        NumericType densityStore = NumericType::FP32;
        NumericType auxStore = NumericType::FP32; // 临时/梯度/累加器等
        NumericType renderTransfer = NumericType::FP32; // 提交渲染的转换目标（保留 float3 默认）

        // ---------------- 核心计算精度（全局默认） ----------------
        NumericType coreCompute = NumericType::FP16; // 主环（邻居/密度/λ/积分）
        bool        forceFp32Accumulate = false;             // 归约/累加是否强制 FP32（数值稳定）
        bool        enableHalfIntrinsics = true;           // 是否启用半精 intrinsic (__hadd2 等)，需架构支持

        // ---------------- 分阶段覆盖（若 useStageOverrides = true 生效） ----------------
        bool        useStageOverrides = true;
        NumericType emissionCompute = NumericType::FP16;
        NumericType gridBuildCompute = NumericType::FP32;
        NumericType neighborCompute = NumericType::FP32;
        NumericType densityCompute = NumericType::FP32;
        NumericType lambdaCompute = NumericType::FP32;
        NumericType integrateCompute = NumericType::FP32;
        NumericType velocityCompute = NumericType::FP32;
        NumericType boundaryCompute = NumericType::FP32;
        NumericType xsphCompute = NumericType::FP32;

        // ---------------- 快捷位掩码控制（当未使用分阶段覆盖时可以批量标记 FP16 计算阶段） ----------------
        uint32_t    fp16StageMask = 0; // 使用 ComputeStageBits，=0 保持全部 FP32

        // ---------------- 自适应精度（预留：根据误差或稳定性动态升降） ----------------
        bool        adaptivePrecision = false;
        float       densityErrorTolerance = 0.01f; // ρ 误差阈值用于判定是否需要提升精度
        float       lambdaVarianceTolerance = 0.05f; // λ 方差阈值
        int         adaptCheckEveryN = 30;    // 多少帧评估一次；<=0 表示禁用评估

        // ---------------- 与旧字段兼容的默认映射策略（无需使用时保持 false） ----------------
        bool        autoMapFromLegacyMixedFlag = false;  // 若 Simulation::useMixedPrecision = true 且用户未手动更改，则可在 BuildSimParams 中做默认映射
    };
    // 统一的运行时控制台（集中所有可调参数）
    struct RuntimeConsole {
        struct App {
            uint32_t width = 1280;
            uint32_t height = 720;
            bool     vsync = false;
            std::string csv_path = "stats.csv";
            // 新增：热重载总开关与频率（帧）
            bool hot_reload = false;
            int  hot_reload_every_n = 0; // 0/负值表示禁用
        } app;

        struct Renderer {
            float clearColor[4] = { 0.1f, 0.2f, 0.35f, 1.0f };
            float particleRadiusPx = 1.0f;
            float thicknessScale = 1.0f;
            float3 eye = make_float3(0.5f, 0.5f, 2.0f);
            float3 at = make_float3(0.5f, 0.5f, 0.5f);
            float3 up = make_float3(0.0f, 1.0f, 0.0f);
            float  fovYDeg = 45.0f;
            float  nearZ = 0.01f;
            float  farZ = 50000.0f;
        } renderer;

        struct Viewer {
            bool  enabled = true;
            float point_size_px = 2.0f;
            float fixed_color[3] = { 0.6f, 0.8f, 1.0f };
            int   draw_subsample = 1;
            float background_color[4] = { 0.02f, 0.02f, 0.02f, 1.0f };
        } viewer;

        // Debug/控制（新增）
        struct Debug {
            bool enabled = true;        // 开启 Debug 模式
            bool pauseOnStart = true;   // 启动即暂停在第 1 帧
            // 采用 Windows VK 与 ASCII 兼容编码（无需包含 windows.h）
            int  keyStep = 32;          // 空格：推进一帧
            int  keyRun = 'R';          // R：连续运行
            int  keyTogglePause = 'P';  // P：在任意时刻切换暂停/继续

            // 日志与诊断开关（新增）
            bool printHotReload = false;        // [HotReload] 提示
            bool printDebugKeys = false;        // [Debug] 键位回显
            bool printPeriodicStats = false;    // [SimStats] 周期统计打印
            bool printSanitize = false;         // [Sanitize] 钳制提示
            bool printWarnings = true;         // [Warn]/[Info] 类一般提示/告警

            // 高开销“塌陷诊断”（包含主机拷贝/近邻统计）
            bool enableAdvancedCollapseDiag = false;
            int  advancedCollapseSampleStride = 16; // >=1，子采样步长

            // —— 数值稳定性专项日志（新增） —__
            bool logStabilityBasic = false;    // 开：输出基础稳定性指标（速度/密度/CFL 等）
            int  logEveryN = 60;               // 每多少帧打印一次（>=1）
            int  logSampleStride = 8;          // 统计核采样步长（>=1）
            int  logMaxHostSample = 65536;     // 允许拷回主机的最大样本数（限制高开销项）

            // 细项开关（在 logStabilityBasic=true 时生效，否则均忽略）
            bool logCFL = true;                // 打印基于 avg/max 速度的 CFL 估算
            bool logLambda = true;             // 打印 PBF λ（约束解）min/max/avg（主机拷样）
            bool logBoundaryApprox = true;     // 估计边界接触占比（主机拷样）
            bool logEnergy = true;             // 打印动能估算与衰减率（主机拷样）
            // 注：logXSphEffect 暂不实现精确估算（需修改核/双缓对比），保留占位以便后续扩展
            bool logXSphEffect = false;        // 占位：XSPH 平滑实际效果估算（未来扩展）

            // —— 新增：统一约束零散日志的分类开关 —__
            bool printDiagnostics = false;     // [Diag] 诊断与网格占用直方等
            bool printHints = false;           // [Hint] 启发式建议
            bool printErrors = true;           // 错误/致命提示（默认开）

            // 新增：诊断/统计的大频率节流（帧）；0/负值表示禁用
            int  diag_every_n = 0;
        } debug;

        // 仿真配置（集中所有物理与发射/域参数）
        struct Simulation {
            // 粒子与发射器
            uint32_t numParticles = 400000;
            uint32_t maxParticles = 2000000;
            uint32_t emitPerStep = 50;
            bool     faucetFillEnable = true;
            bool     recycleToNozzle = false;
            float3   nozzlePos = make_float3(100.0f, 199.0f, 100.0f);
            float3   nozzleDir = make_float3(0.0f, -1.0f, 0.0f);
            float    nozzleRadius = 20.0f;
            float    nozzleSpeed = 50.0f;
            float    recycleYOffset = 1e-3f;

            // Poisson-disk 最小间距（相对 h）
            float    poisson_min_spacing_factor_h = 0.8f;

            // —— 新增：初始/发射抖动（用于打破规则立方体格点导致的“团块不散”） —— 
            // 初始格点随机扰动开关（seedBoxLattice / seedBoxLatticeAuto 后应用）
            bool     initial_jitter_enable = true;
            // 扰动幅度 = initial_jitter_scale_h * h （若 h 不可用则退化为 spacing）
            float    initial_jitter_scale_h = 0.01f;
            uint32_t initial_jitter_seed = 0xC0FFEEu;
            // 发射粒子附加扰动（在喷口圆盘内 Poisson / 随机分布基础上再微扰）
            bool     emit_jitter_enable = false;
            float    emit_jitter_scale_h = 0.005f;
            uint32_t emit_jitter_seed = 0xABCDEFu;

            // 基本动力学
            float    dt = 0.0167f;
            float    cfl = 0.45f;
            float3   gravity = make_float3(0.0f, -9.8f, 0.0f);
            float    restDensity = 1.0f;

            // 质量定义
            enum class MassMode : uint32_t { Explicit = 0, SphereByRadius = 1, UniformLattice = 2 };
            MassMode massMode = MassMode::UniformLattice;
            float    lattice_spacing_factor_h = 1.0f;
            float    particleMass = 1.0f;
            float    particleVolumeScale = 1.0f;

            // 物理几何定义
            float    particleRadiusWorld = 1.0f;
            bool     deriveHFromRadius = true;
            float    h_over_r = 2.0f;

            // PBF 调参
            sim::PbfTuning pbf{};

            // XSPH 系数
            float    xsph_c = 0.05f;

            // 邻域核参数
            float    smoothingRadius = 2.0f;
            float3   gridMins = make_float3(0.0f, 0.0f, 0.0f);
            float3   gridMaxs = make_float3(200.0f, 200.0f, 200.0f);
            float    cellSize = 0.0f;

            // 其他数值参数
            int      solverIters = 2;
            int      maxNeighbors = 64;
            bool     useMixedPrecision = true;
            int      sortEveryN = 4;
            float    boundaryRestitution = 0.0f;

            // —— 新增：自适应 h/喷口半径（降低稀疏场景下“无邻居”概率） —__
            bool     auto_tune_h = true;     // 开启基于目标邻居数的 h 估算
            int      target_neighbors = 30;  // 目标邻居数（期望范围 20~60）
            float    h_min = 0.25f;          // h 下限（避免过小）
            float    h_max = 1e3f;           // h 上限防失控
            float    nozzle_radius_factor_h = 1.5f; // 喷口半径上限系数：min(nozzleRadius, factor*h)
            // ================== 新增：Demo 模式选择 ==================
            enum class DemoMode : uint32_t { Faucet = 0, CubeMix = 1 };
            DemoMode demoMode = DemoMode::CubeMix; // 切换：水龙头 / 立方体混合

            // ================== 新增：立方体混合参数 ==================
            // 是否根据 numParticles 自动分解为 cube_group_count * (cube_edge_particles^3)
            bool     cube_auto_partition = false;
            // 手动指定立方体数量（粒子团数量），仅在 cube_auto_partition=false 时使用
            uint32_t cube_group_count = 128;
            // 单个立方体边长（按粒子数，边上粒子个数）；用于 cube_edge_particles^3
            uint32_t cube_edge_particles = 20;
            // 最大允许粒子团数量（用于颜色数组与安全限制）
            static constexpr uint32_t cube_group_count_max = 512;

            // 立方体层数（垂直分层放置）。若 cube_group_count=32 且 cube_layers=2 -> 每层16个
            uint32_t cube_layers = 8;

            // 立方体中心之间的水平间距（世界单位，沿 X/Z）
            float    cube_group_spacing_world = 80.0f;
            // 层之间的垂直间距（世界单位，立方体 Y 方向层距）
            float    cube_layer_spacing_world = 200.0f;

            // 立方体底层离地高度（世界坐标 Y）
            float    cube_base_height = 50.0f;

            // 单个立方体内格点的粒子间距缩放（相对 smoothingRadius 或 h），用于调节初始紧实程度
            float    cube_lattice_spacing_factor_h = 1.05f;

            // 立方体粒子团的初始密度（若需要与全局 restDensity 不同，可用于初始化时的质量或体积标定）
            float    cube_initial_density = 1.0f;

            // 颜色相关：是否为每个粒子团分配随机亮色
            bool     cube_color_enable = true;
            uint32_t cube_color_seed = 0xBADC0DEu;
            // RGB 最低亮度（近似：max(r,g,b) 或简单使用每分量下限）
            float    cube_color_min_component = 0.15f;
            // 颜色最小欧氏距离阈值（避免相邻过相似）
            float    cube_color_min_distance = 0.25f;
            // 是否启用“相邻立方体避免相似”逻辑（基于网格邻接判断）
            bool     cube_color_avoid_adjacent_similarity = true;

            // 生成颜色失败时的重试上限（避免死循环）
            int      cube_color_retry_max = 64;

            // 是否在根据布置结果动态缩放 / 调整 gridMaxs
            bool     cube_auto_fit_domain = true;
            // 域扩展边界余量（相对排布包围盒的比例）
            float    cube_domain_margin_scale = 1.2f;

            // 是否应用 initial_jitter_* 到立方体粒子初始化（与水龙头播种逻辑分离控制）
            bool     cube_apply_initial_jitter = true;

            // 若需要对每层 Y 偏移做附加随机扰动（打散层间绝对平面）
            bool     cube_layer_jitter_enable = false;
            float    cube_layer_jitter_scale_h = 0.05f;
            uint32_t cube_layer_jitter_seed = 0x13579BDFu;

            // 预留：存储为每个粒子团生成的颜色（RGB）。运行时填充。
            // 注意：实际使用时请限制访问范围 <= 实际 cube_group_count
            float    cube_group_colors[cube_group_count_max][3] = {};

            // 预留：分组粒子数（每团 edge^3），自动分解时写入，便于外部初始化阶段引用
            uint32_t cube_particles_per_group = 0;
            PrecisionConfig precision{};

            // ======= 新增：针对高 dt 的稳定性增强开关 =======
            // 半隐式积分：true 使用 v(t+dt)=v(t)+g*dt 然后 x(t+dt)=x(t)+v(t+dt)*dt
            bool     integrate_semi_implicit = true;
            // λ Warm-Start：保留上一帧 λ 并按衰减系数衰减；新发射粒子置 0
            bool     lambda_warm_start_enable = true;
            float    lambda_warm_start_decay = 0.5f; // 0~1，典型 0.3~0.7
            // XPBD 合规：开启后使用 (C + α*λ_prev)/(Σgrad^2 + α)，α = compliance / dt^2
            bool     xpbd_enable = true;
            float    xpbd_compliance = 2e-5f; // 典型区间 1e-5 ~ 5e-5


        } sim;

        struct Performance {
            float grid_cell_size_multiplier = 1.0f;
            int   neighbor_cap = 0;
            int   launch_bounds_tbs = 256;
            int   min_blocks_per_sm = 2;
            bool  use_cuda_graphs = true;

            // 新增：启用哈希/压缩网格（按 cell-key 排序 + 压缩段表）
            bool  use_hashed_grid = true;
            // 新增：压缩段表重建频率（帧）：>=1
            int   sort_compact_every_n = 8;
            // 新增：邻域查段方式（true=对压缩 key 做二分；false=使用辅表哈希，后续扩展）
            bool  compact_binary_search = true;
            // 新增：打印网格统计（非空 cell 数、均值/最大 occupancy 等）
            bool  log_grid_compact_stats = false;
            // 新增：NVTX 总开关（运行时控制
            bool  enable_nvtx = true;
            // 新增：参数更新最小间隔（帧），避免频繁 GraphCapture；<=0 表示禁用节流
            int   graph_param_update_min_interval = 30;
            // 新增：帧级 GPU 计时采样频率（帧）。<=0 禁用 cudaEventRecord
            int   frame_timing_every_n = 30;
            // ===== 新增：Graph 指针热更新开关 =====
            // false: 使用原始末尾 cudaMemcpyAsync 拷贝 (pos_pred->pos, delta->vel)
            // true : 使用 ping-pong + cudaGraphExecKernelNodeSetParams (零拷贝模式)
            bool  graph_hot_update_enable = true;
            // 可选：扫描 kernel 参数槽位上限，限制 patch 遍历成本（默认 64）
            int   graph_hot_update_scan_limit = 4096;
			// 新增：允许使用 ping-pong 机制（前提是外部未绑定预测位置缓冲）
            bool allow_pingpong_with_external_pred = true;
            // 新增：是否启用双外部位置缓冲零拷贝 ping-pong（否则使用单外部预测缓冲旧实现）
            bool use_external_pos_pingpong = true;
            // 新增：是否仍然等待渲染 GPU 完成（保持旧行为）
            bool  wait_render_gpu = false;
        } perf;
    };

    RuntimeConsole& Instance();

    void BuildSimParams(const RuntimeConsole& c, sim::SimParams& out);
    void BuildDeviceParams(const RuntimeConsole& c, sim::DeviceParams& out);
    void BuildEmitParams(const RuntimeConsole& c, sim::EmitParams& out);
    void BuildRenderInitParams(const RuntimeConsole& c, gfx::RenderInitParams& out);
    void ApplyRendererRuntime(const RuntimeConsole& c, gfx::RendererD3D12& r);
    void FitCameraToDomain(RuntimeConsole& c);
    // —— 新增：CubeMix 准备（自动分解 + 域拟合 + 颜色） —— //
    void PrepareCubeMix(RuntimeConsole& c);

    // —— 新增：生成立方体中心列表（供播种使用，与 PrepareCubeMix 保持一致） —— //
    void GenerateCubeMixCenters(const RuntimeConsole& c, std::vector<float3>& outCenters);


} // namespace console