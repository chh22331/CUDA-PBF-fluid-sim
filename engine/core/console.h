#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h> // for float3/make_float3, int3/make_int3
#include "../../sim/parameters.h"
#include "../../engine/gfx/renderer.h"

namespace console {

    // 新增：强制使用 CUDA Graph（阶段一精简：删除可选开关）
    inline constexpr bool AlwaysUseCudaGraphs = true;

    struct RuntimeConsole {
        struct App {
            uint32_t width = 1280;
            uint32_t height = 720;
            bool     vsync = false;
            std::string csv_path = "stats.csv";
            bool hot_reload = false;
            int  hot_reload_every_n = 0;
        } app;

        struct Renderer {
            float clearColor[4] = { 0.1f, 0.2f, 0.35f, 1.0f };
            float particleRadiusPx = 1.0f;
            float thicknessScale = 1.0f;
            float3 eye = make_float3(350.0f, 50.0f, 300.0f);
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

        struct Debug {
            bool enabled = false;
            bool pauseOnStart = true;
            int  keyStep = 32;
            int  keyRun = 'R';
            int  keyTogglePause = 'P';

            bool printHotReload = false;
            bool printDebugKeys = false;
            bool printPeriodicStats = false;
            bool printSanitize = false;
            bool printWarnings = true;

            bool enableAdvancedCollapseDiag = false;
            int  advancedCollapseSampleStride = 16;

            bool logStabilityBasic = false;
            int  logEveryN = 60;
            int  logSampleStride = 8;
            int  logMaxHostSample = 65536;

            bool logCFL = true;
            bool logLambda = true;
            bool logBoundaryApprox = true;
            bool logEnergy = true;
            bool logXSphEffect = false;

            bool printDiagnostics = false;
            bool printHints = false;
            bool printErrors = true;

            int  diag_every_n = 0;
        } debug;

        struct Simulation {
            uint32_t numParticles = 400000;
            uint32_t maxParticles = 2000000;
            uint32_t emitPerStep = 50;
  
            float    poisson_min_spacing_factor_h = 0.8f;

            float    dt = 0.0167f;
            float    cfl = 0.45f;
            float3   gravity = make_float3(0.0f, -9.8f, 0.0f);
            float    restDensity = 1.0f;

            enum class MassMode : uint32_t { Explicit = 0, SphereByRadius = 1, UniformLattice = 2 };
            MassMode massMode = MassMode::UniformLattice;
            float    lattice_spacing_factor_h = 1.0f;
            float    particleMass = 1.0f;
            float    particleVolumeScale = 1.0f;

            float    particleRadiusWorld = 1.0f;
            bool     deriveHFromRadius = true;
            float    h_over_r = 2.5f;

            sim::PbfTuning pbf{};
            float    xsph_c = 0.05f;

            float    smoothingRadius = 2.0f;
            float3   gridMins = make_float3(0.0f, 0.0f, 0.0f);
            float3   gridMaxs = make_float3(200.0f, 200.0f, 200.0f);
            float    cellSize = 0.0f;

            int      solverIters = 1;
            int      maxNeighbors = 64;
            bool     useMixedPrecision = true;
            int      sortEveryN = 4;
            float    boundaryRestitution = 0.0f;

            bool     auto_tune_h = true;
            int      target_neighbors = 30;
            float    h_min = 0.25f;
            float    h_max = 1e3f;
            float    nozzle_radius_factor_h = 1.5f;

            enum class DemoMode : uint32_t {CubeMix = 1 };
            DemoMode demoMode = DemoMode::CubeMix;

            bool     cube_auto_partition = false;
            uint32_t cube_group_count = 128;
            uint32_t cube_edge_particles = 23;
            static constexpr uint32_t cube_group_count_max = 512;
            uint32_t cube_layers = 8;
            float    cube_group_spacing_world = 80.0f;
            float    cube_layer_spacing_world = 100.0f;
            float    cube_base_height = 100.0f;
            float    cube_lattice_spacing_factor_h = 1.02f;
            float    cube_initial_density = 1.0f;

            bool     cube_color_enable = true;
            uint32_t cube_color_seed = 0xBADC0DEu;
            float    cube_color_min_component = 0.15f;
            float    cube_color_min_distance = 0.25f;
            bool     cube_color_avoid_adjacent_similarity = true;
            int      cube_color_retry_max = 64;

            bool     cube_auto_fit_domain = true;
            float    cube_domain_margin_scale = 1.2f;
            bool     cube_apply_initial_jitter = true;
            bool     cube_layer_jitter_enable = false;
            float    cube_layer_jitter_scale_h = 0.05f;
            uint32_t cube_layer_jitter_seed = 0x13579BDFu;

            float    cube_group_colors[cube_group_count_max][3] = {};
            uint32_t cube_particles_per_group = 0;
            
            bool     integrate_semi_implicit = true;
            bool     lambda_warm_start_enable = true;
            float    lambda_warm_start_decay = 0.5f;
            bool     xpbd_enable = true;
            float    xpbd_compliance = 2e-5f;

            bool     initial_jitter_enable = true;
            float    initial_jitter_scale_h = 0.01f;
            uint32_t initial_jitter_seed = 0xC0FFEEu;
        } sim;

        struct Performance {
            float grid_cell_size_multiplier = 1.0f;
            int   neighbor_cap = 0;
            int   launch_bounds_tbs = 256;
            int   min_blocks_per_sm = 2;

            bool  use_hashed_grid = false;
            int   sort_compact_every_n = 1;
            bool  compact_binary_search = true;
            bool  log_grid_compact_stats = false;
            bool  enable_nvtx = true;
            int   graph_param_update_min_interval = 30;
            int   frame_timing_every_n = 30;
        } perf;

        struct Benchmark {
            bool enabled = true;
            int total_frames = 0;
            double total_seconds = 40;
            int sample_begin_frame = 0;
            int sample_end_frame = -1;
            bool use_time_range = true;
            double sample_begin_seconds = 15.0;
            double sample_end_seconds = -1.0;
            bool print_per_frame = false;
        } bench;
    };

    RuntimeConsole& Instance();

    void BuildSimParams(const RuntimeConsole& c, sim::SimParams& out);
    void BuildDeviceParams(const RuntimeConsole& c, sim::DeviceParams& out);
    void BuildRenderInitParams(const RuntimeConsole& c, gfx::RenderInitParams& out);
    void ApplyRendererRuntime(const RuntimeConsole& c, gfx::RendererD3D12& r);
    void PrepareCubeMix(RuntimeConsole& c);
    void GenerateCubeMixCenters(const RuntimeConsole& c, std::vector<float3>& outCenters);
} // namespace console