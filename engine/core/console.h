#pragma once
#include <cstdint>
#include <string>
#include <cuda_runtime.h> // for float3/make_float3, int3/make_int3
#include "../../sim/parameters.h"
#include "../../engine/gfx/renderer.h"
#include "../../sim/emit_params.h"

namespace console {

    // 统一的运行时控制台（集中所有可调参数）
    struct RuntimeConsole {
        struct App {
            uint32_t width = 1280;
            uint32_t height = 720;
            bool     vsync = false;
            std::string csv_path = "stats.csv";
        } app;

        struct Renderer {
            float clearColor[4] = { 0.1f, 0.2f, 0.35f, 1.0f };
            float particleRadiusPx = 3.0f;
            float thicknessScale = 1.0f;
            float3 eye = make_float3(0.5f, 0.5f, 2.0f);
            float3 at  = make_float3(0.5f, 0.5f, 0.5f);
            float3 up  = make_float3(0.0f, 1.0f, 0.0f);
            float  fovYDeg = 45.0f;
            float  nearZ = 0.01f;
            float  farZ  = 1000.0f;
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
            bool printWarnings = false;          // [Warn]/[Info] 类一般提示/告警

            // 高开销“塌陷诊断”（包含主机拷贝/近邻统计）
            bool enableAdvancedCollapseDiag = false;
            int  advancedCollapseSampleStride = 16; // >=1，子采样步长
        } debug;

        // 仿真配置（集中所有物理与发射/域参数）
        struct Simulation {
            // 粒子与发射器
            uint32_t numParticles = 1;
            uint32_t maxParticles = 20000;
            uint32_t emitPerStep = 20;
            bool     faucetFillEnable = true;
            bool     recycleToNozzle = false;
            float3   nozzlePos   = make_float3(25.0f, 49.0f, 25.0f);
            float3   nozzleDir   = make_float3(0.0f, -1.0f, 0.0f);
            // 缩回与 h 同阶的喷口尺度
            float    nozzleRadius= 20.0f;
            float    nozzleSpeed = 50.0f;
            float    recycleYOffset = 1e-3f;

            // Poisson-disk 发射的最小间距（相对 h 的倍数）
            float    poisson_min_spacing_factor_h = 0.8f;

            // 基本动力学
            float    dt = 0.004f;
            float    cfl = 0.45f;
            float3   gravity = make_float3(0.0f, -9.8f, 0.0f);
            float    restDensity = 1.0f;

            // 质量定义（完全由 console 统一下发）
            enum class MassMode : uint32_t { Explicit = 0, SphereByRadius = 1, UniformLattice = 2 };
            MassMode massMode = MassMode::Explicit;
            float    lattice_spacing_factor_h = 1.0f;
            float    particleMass = 1.0f;     
            float    particleVolumeScale = 1.0f; 

            // 物理几何定义
            float    particleRadiusWorld = 1.0f;
            bool     deriveHFromRadius = true;
            float    h_over_r = 2.0f;

            // PBF 调参
            sim::PbfTuning pbf{};

            // XSPH 系数（移动到 console：<=0 关闭，常用 0.01~0.1）
            float    xsph_c = 0.05f;

            // 邻域核参数
            float    smoothingRadius = 2.0f;
            float3   gridMins = make_float3(0.0f, 0.0f, 0.0f);
            float3   gridMaxs = make_float3(50.0f, 50.0f, 50.0f);
            float    cellSize = 0.0f;

            // 其他数值参数
            int      solverIters = 4;
            int      maxNeighbors = 16384;
            bool     useMixedPrecision = true;
            int      sortEveryN = 1;
            float    boundaryRestitution = 0.0f;
        } sim;

        struct Performance {
            float grid_cell_size_multiplier = 1.0f;
            int   neighbor_cap = 0;          
            int   launch_bounds_tbs = 256;
            int   min_blocks_per_sm = 2;
            bool  use_cuda_graphs = true;
        } perf;
    };

    RuntimeConsole& Instance();

    void BuildSimParams(const RuntimeConsole& c, sim::SimParams& out);
    void BuildDeviceParams(const RuntimeConsole& c, sim::DeviceParams& out);
    void BuildEmitParams(const RuntimeConsole& c, sim::EmitParams& out);
    void BuildRenderInitParams(const RuntimeConsole& c, gfx::RenderInitParams& out);
    void ApplyRendererRuntime(const RuntimeConsole& c, gfx::RendererD3D12& r);
    void FitCameraToDomain(RuntimeConsole& c);

} // namespace console