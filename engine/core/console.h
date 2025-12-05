#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h> // for float3/make_float3, int3/make_int3
#include "../../sim/parameters.h"
#include "../../engine/gfx/renderer.h"

namespace console {

/*
    Runtime configuration and simple helpers used by the standalone app and
    the renderer. This header centralizes user-tweakable parameters that drive
    simulation, rendering, debugging and performance tuning. Values here
    are copied into sim::SimParams / sim::DeviceParams and into renderer
    initialization structures at startup or when hot-reloading settings.

    Notes:
    - Keep this file header-only and POD-like so it can be easily serialized
      or inspected by a debug UI.
    - Naming follows the project convention: types PascalCase, functions
      camelCase, members prefixed with m_ where used elsewhere. This file
      uses plain member names for clarity in the runtime console struct.
*/

// Force CUDA Graphs for the application. Kept as a compile-time constant to
// make behavior deterministic and to simplify the control flow elsewhere.
// Stage one: no runtime toggle -- graphs are always used when this is true.
inline constexpr bool AlwaysUseCudaGraphs = true;

/*
    RuntimeConsole
    --------------
    A single aggregated structure containing all runtime-configurable
    parameters grouped by purpose. An instance is accessed via Instance()
    and can be used to populate sim::SimParams, sim::DeviceParams and the
    renderer init parameters.

    The design intent:
    - Keep configuration grouped to make UI/hot-reload integration straightforward.
    - Store reasonable defaults for interactive usage.
*/
struct RuntimeConsole {
    // Application-level settings (window, vsync, hot reload, CSV logging)
    struct App {
        uint32_t width = 1280;                 // initial window width (pixels)
        uint32_t height = 720;                 // initial window height (pixels)
        bool     vsync = false;                // enable vertical sync
        std::string csv_path = "stats.csv";    // path to write CSV benchmark/stats
        bool hot_reload = false;               // enable hot-reload of shaders/config
        int  hot_reload_every_n = 0;           // frames between hot-reload attempts (0 = disabled)
        bool frame_cap_enabled = true;        // limit frame rate when true
        int  frame_cap_fps = 60;              // target max FPS when frame cap is enabled
    } app;

    // Renderer tuning and camera defaults.
    struct Renderer {
        // Clear color used by the GPU clear pass (RGBA).
        float clearColor[4] = { 0.1f, 0.2f, 0.35f, 1.0f };

        // Particle thickness/radius used by the rasterizer in pixel units.
        float particleRadiusPx = 1.0f;

        // Generic thickness scale. Historically used for thickness and
        // as a normalization factor for speed coloring. This value may be
        // overridden by ApplyRendererRuntime when speedColorAutoScale is false.
        float thicknessScale = 1.0f;

        // Default camera transform (eye, look-at, up).
        float3 eye = make_float3(350.0f, 50.0f, 300.0f);
        float3 at  = make_float3(0.5f, 0.5f, 0.5f);
        float3 up  = make_float3(0.0f, 1.0f, 0.0f);

        // Projection parameters.
        float  fovYDeg = 45.0f;
        float  nearZ = 0.01f;
        float  farZ = 50000.0f;

        // Render mode: maps directly to gfx::RendererD3D12::RenderMode.
        // Default shows speed-based coloring.
        gfx::RendererD3D12::RenderMode renderMode = gfx::RendererD3D12::RenderMode::SpeedColor;

        // Controls automatic scaling of speed-based color mapping.
        // If true, thicknessScale is treated as the color normalization scale.
        // If false, thicknessScale will be set to 1.0f / speedColorMaxSpeedHint in
        // ApplyRendererRuntime so the color factor = speed * (1/maxSpeedHint).
        bool  speedColorAutoScale = true;
        float speedColorMaxSpeedHint = 30.0f; // hint for manual speed normalization
    } renderer;

    // Viewer-specific drawing options that are independent from the GPU renderer.
    struct Viewer {
        bool  enabled = true;                  // toggle viewer overlay
        float point_size_px = 2.0f;            // point primitive size in pixels
        float fixed_color[3] = { 0.6f, 0.8f, 1.0f }; // fallback color for simple rendering
        int   draw_subsample = 1;              // draw every Nth particle (for debugging)
        float background_color[4] = { 0.02f, 0.02f, 0.02f, 1.0f }; // UI background
    } viewer;

    // Debugging and logging controls.
    struct Debug {
        bool enabled = false;                  // pause mode
        bool pauseOnStart = true;              // pause simulation at startup
        int  keyStep = 32;                     // key for stepping frames
        int  keyRun = 'R';                     // key to run
        int  keyTogglePause = 'P';             // toggle pause key

        bool printHotReload = false;           // log hot-reload actions
        bool printSanitize = false;            // print data sanitize warnings
        bool printWarnings = true;             // enable general warnings

        // Advanced diagnostics for collapse/instability investigation.
        bool enableAdvancedCollapseDiag = false;
        int  advancedCollapseSampleStride = 16;

        bool printDiagnostics = false;
        bool printHints = false;
        bool printErrors = true;
    } debug;

    // Simulation parameters that map to sim::SimParams and tuning structures.
    struct Simulation {
        Simulation();
        void refreshCubeMixDerived();

        uint32_t numParticles = 800000;        // just ignored it
        uint32_t maxParticles = 10000000;      // hard cap for allocation
        uint32_t emitPerStep = 50;             // particles emitted per simulation step

        // Poisson-disk minimum spacing relative to smoothing length h.
        float    poisson_min_spacing_factor_h = 0.8f;

        // Time stepping and CFL control.
        float    dt = 0.016667f;
        float    cfl = 0.45f;

        // Global physics parameters.
        float3   gravity = make_float3(0.0f, -9.8f, 0.0f);
        float    restDensity = 1.0f;

        // Mass handling modes for particles:
        // - Explicit: mass provided directly
        // - SphereByRadius: derive mass from radius assuming sphere density
        // - UniformLattice: constant mass from lattice spacing
        enum class MassMode : uint32_t { Explicit = 0, SphereByRadius = 1, UniformLattice = 2 };
        MassMode massMode = MassMode::UniformLattice;

        float    lattice_spacing_factor_h = 1.0f; // scaling factor when using lattice mass mode
        float    particleMass = 1.0f;            // explicit particle mass when used
        float    particleVolumeScale = 1.0f;     // scale applied to computed particle volume

        // Particle sizing expressed in world units and smoothing relation.
        float    particleRadiusWorld = 1.0f;
        bool     deriveHFromRadius = true;       // derive smoothing radius h from particle radius
        float    h_over_r = 2.5f;                // ratio h / radius

        // PBF tuning block (see sim::PbfTuning for available fields).
        sim::PbfTuning pbf{};
        float    xsph_c = 0.05f;                 // XSPH velocity correction coefficient

        // Kernel / neighborhood settings.
        float    smoothingRadius = 3.5f;
        float3   gridMins = make_float3(0.0f, 0.0f, 0.0f);
        float3   gridMaxs = make_float3(200.0f, 200.0f, 200.0f);
        float    cellSize = 0.0f;                // computed cell size (updated at runtime)

        int      solverIters = 1;                // number of constraint solver iterations
        int      maxNeighbors = 64;              // neighbor sample cap
        float    boundaryRestitution = 0.0f;

        // Auto-tuning of smoothing length to maintain target_neighbors.
        bool     auto_tune_h = true;
        int      target_neighbors = 30;
        float    h_min = 0.25f;
        float    h_max = 1e3f;
        float    nozzle_radius_factor_h = 1.5f;

        // Demo presets and parameters (CubeMix used by default).
        enum class DemoMode : uint32_t {CubeMix = 1 };
        DemoMode demoMode = DemoMode::CubeMix;

        // CubeMix-specific parameters for batched group emit/placement.
        // These control the number of cube groups and particles.
        bool     cube_auto_partition = false;
        uint32_t cube_group_count = 8;
        uint32_t cube_edge_particles = 60;
        static constexpr uint32_t cube_group_count_max = 512;
        uint32_t cube_layers = 2;
        float    cube_group_spacing_world = 200.0f;
        float    cube_layer_spacing_world = 200.0f;
        float    cube_base_height = 200.0f;
        float    cube_lattice_spacing_factor_h = 1.02f;
        float    cube_initial_density = 1.0f;

        // Color and seeding options for grouping visualization.
        bool     cube_color_enable = true;
        uint32_t cube_color_seed = 0xBADC0DEu;
        float    cube_color_min_component = 0.15f;
        float    cube_color_min_distance = 0.25f;
        bool     cube_color_avoid_adjacent_similarity = true;
        int      cube_color_retry_max = 64;

        bool     cube_auto_fit_domain = true;    // expand domain to fit cube groups
        float    cube_domain_margin_scale = 1.2f;
        bool     cube_apply_initial_jitter = true;
        bool     cube_layer_jitter_enable = false;
        float    cube_layer_jitter_scale_h = 0.05f;
        uint32_t cube_layer_jitter_seed = 0x13579BDFu;

        // Precomputed per-group color storage. The array is sized to the
        // compile-time cube_group_count_max to avoid heap allocations for UI use.
        float    cube_group_colors[cube_group_count_max][3] = {};
        uint32_t cube_particles_per_group = 0;

        // Integration and solver warm-starting options.
        bool     integrate_semi_implicit = true;
        bool     lambda_warm_start_enable = true;
        float    lambda_warm_start_decay = 0.5f;
        bool     xpbd_enable = true;
        float    xpbd_compliance = 2e-5f;

        // Initial jitter for particle placement to prevent perfect lattice artifacts.
        bool     initial_jitter_enable = true;
        float    initial_jitter_scale_h = 0.01f;
        uint32_t initial_jitter_seed = 0xC0FFEEu;
    } sim;

    // Performance tuning knobs. These affect grid sizing, kernel launches,
    // sorting/compaction behavior and graph update heuristics.
    struct Performance {
        float grid_cell_size_multiplier = 1.0f; // scale applied to computed grid cell size
        int   neighbor_cap = 0;                 // clamp on neighbor count (0 = no clamp)
        int   launch_bounds_tbs = 256;          // thread-block size hint for kernels
        int   min_blocks_per_sm = 2;            // occupancy hint

        bool  use_hashed_grid = false;          // use hashed grid instead of dense grid
        int   sort_compact_every_n = 1;         // frequency of sort/compact operations
        bool  compact_binary_search = true;     // use binary search during compaction
        bool  log_grid_compact_stats = false;   // dump compaction stats
        bool  enable_nvtx = true;               // annotate NVTX ranges when available
        int   graph_param_update_min_interval = 30; // min frames between graph param updates
        int   frame_timing_every_n = 30;        // frame sampling interval for timing
    } perf;

    // Benchmarking options used by the harness to run time-limited measurements.
    struct Benchmark {
        bool enabled = false;                    // run benchmark mode automatically
        int total_frames = 0;                   // number of frames to collect (0 = use time)
        double total_seconds = 40;              // duration in seconds (when frames == 0)
        int sample_begin_frame = 0;             // start frame to include in samples
        int sample_end_frame = -1;              // end frame to include (inclusive), -1 = until end
        bool use_time_range = true;             // use seconds range instead of frames
        double sample_begin_seconds = 15.0;     // begin time for sampling within run
        double sample_end_seconds = -1.0;       // end time for sampling, -1 = until finish
        bool print_per_frame = false;           // optionally print per-frame stats during sampling
    } bench;
};

// Returns global RuntimeConsole instance. The single instance is intended to be
// the authoritative source for runtime parameters and is used during startup to
// construct sim::SimParams, sim::DeviceParams and renderer init data.
RuntimeConsole& Instance();

/*
    Utility functions that copy values from the RuntimeConsole into the
    various engine/config structs expected by the simulation and renderer.

    - BuildSimParams: fills a sim::SimParams from console state.
    - BuildDeviceParams: fills a sim::DeviceParams for GPU-side configuration.
    - BuildRenderInitParams: converts renderer settings into gfx::RenderInitParams.
    - ApplyRendererRuntime: applies dynamic renderer runtime tweaks (e.g. color scaling).
    - PrepareCubeMix: precomputes or adjusts parameters for the CubeMix demo.
    - GenerateCubeMixCenters: produces group center positions for CubeMix initialization.

    Implementations live in the corresponding .cpp file and must keep mapping
    logic consistent with the sim and gfx parameter semantics.
*/
void BuildSimParams(const RuntimeConsole& c, sim::SimParams& out);
void BuildDeviceParams(const RuntimeConsole& c, sim::DeviceParams& out);
void BuildRenderInitParams(const RuntimeConsole& c, gfx::RenderInitParams& out);
void ApplyRendererRuntime(const RuntimeConsole& c, gfx::RendererD3D12& r);
void PrepareCubeMix(RuntimeConsole& c);
void GenerateCubeMixCenters(const RuntimeConsole& c, std::vector<float3>& outCenters);

} // namespace console
