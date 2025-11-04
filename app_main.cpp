#define NOMINMAX
#include <windows.h>
#include <windowsx.h>

#include <cstdio>
#include <cwchar>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <string>
#include <algorithm>

#include "engine/core/config.h"
#include "engine/core/profiler.h"
#include "engine/gfx/renderer.h"
#include "engine/core/console.h"
#include "engine/core/prof_nvtx.h"
#include "sim/simulator.h"
#include "sim/parameters.h"
#include "sim/stats.h"

// ---------------- Debug 全局状态 ----------------
static bool g_DebugEnabled = false;
static bool g_DebugPaused = false;
static bool g_DebugStepRequested = false;

static inline bool KeyDown(int vk) {
    return (GetAsyncKeyState(vk) & 0x8000) != 0;
}

static bool KeyPressedOnce(int vk) {
    static uint8_t prev[512] = {};
    vk &= 0x1FF;
    bool down = KeyDown(vk);
    bool fired = down && (prev[vk] == 0);
    prev[vk] = down ? 1 : 0;
    return fired;
}

static void UpdateWindowTitleHUD(HWND hwnd, uint32_t particleCount, double fps) {
    wchar_t buf[256];
    if (fps < 1e-3) fps = 0.0;
    // 使用符合 ISO 的 swprintf 版本，避免 C4996
    swprintf(buf, _countof(buf), L"PBF-X | N=%u | FPS=%.1f", particleCount, fps);
    SetWindowTextW(hwnd, buf);
}

// ---------------- 简易 float3 工具 ----------------
static inline float3 f3_make(float x, float y, float z) { return make_float3(x, y, z); }
static inline float3 f3_add(const float3& a, const float3& b) { return f3_make(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline float3 f3_sub(const float3& a, const float3& b) { return f3_make(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline float3 f3_scale(const float3& a, float s) { return f3_make(a.x * s, a.y * s, a.z * s); }
static inline float  f3_dot(const float3& a, const float3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline float3 f3_cross(const float3& a, const float3& b) {
    return f3_make(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static inline float  f3_len(const float3& a) { return std::sqrt(f3_dot(a, a)); }
static inline float3 f3_norm(const float3& a) {
    float l = f3_len(a);
    return (l > 1e-20f) ? f3_scale(a, 1.0f / l) : f3_make(0, 0, 0);
}

// ---------------- 自由飞行摄像机 ----------------
struct FreeFlyCamera {
    float3 position{};
    double yaw = 0.0;
    double pitch = 0.0;
    float  moveSpeed = 1.0f;
    float  mouseSensitivity = 0.002f;
};
static FreeFlyCamera g_Camera;
static FreeFlyCamera* g_cam = nullptr;
static bool  g_RmbDown = false;
static POINT g_LastMouse{};
static float g_PendingWheel = 0.0f;

static inline float3 CamForward(const FreeFlyCamera& c) {
    float cp = std::cos((float)c.pitch);
    float sp = std::sin((float)c.pitch);
    float cy = std::cos((float)c.yaw);
    float sy = std::sin((float)c.yaw);
    return f3_norm(f3_make(sy * cp, sp, cy * cp));
}

static inline void CamBasis(const FreeFlyCamera& c, float3& fwd, float3& right, float3& up) {
    const float3 worldUp = f3_make(0, 1, 0);
    fwd = CamForward(c);
    right = f3_norm(f3_cross(fwd, worldUp));
    up = f3_cross(right, fwd);
}

static FreeFlyCamera MakeCameraFromCc(const console::RuntimeConsole& cc) {
    FreeFlyCamera cam{};
    cam.position = cc.renderer.eye;
    float3 fwd = f3_norm(f3_sub(cc.renderer.at, cc.renderer.eye));
    cam.pitch = std::asin(std::clamp((double)fwd.y, -1.0, 1.0));
    cam.yaw = std::atan2((double)fwd.x, (double)fwd.z);
    cam.moveSpeed = 1.0f;
    return cam;
}

static float ComputeBaseSpeedFromDomain(const float3& mins, const float3& maxs) {
    float3 ext = f3_sub(maxs, mins);
    float diag = f3_len(ext);
    return std::max(0.3f, diag * 0.3f);
}

static void UpdateFreeFlyCamera(FreeFlyCamera& cam, float dtSec) {
    float3 fwd, right, up;
    CamBasis(cam, fwd, right, up);

    if (std::abs(g_PendingWheel) > 1e-4f) {
        float dolly = g_PendingWheel * 0.0025f * cam.moveSpeed;
        cam.position = f3_add(cam.position, f3_scale(fwd, dolly));
        g_PendingWheel = 0.0f;
    }

    auto down = [](int vk) { return (GetAsyncKeyState(vk) & 0x8000) != 0; };
    float3 move = f3_make(0, 0, 0);
    if (down('W')) move = f3_add(move, fwd);
    if (down('S')) move = f3_sub(move, fwd);
    if (down('D')) move = f3_add(move, right);
    if (down('A')) move = f3_sub(move, right);
    if (down('E')) move = f3_add(move, up);
    if (down('Q')) move = f3_sub(move, up);

    float speedMul = 1.0f;
    if (down(VK_SHIFT))   speedMul *= 5.0f;
    if (down(VK_CONTROL)) speedMul *= 0.2f;

    if (f3_len(move) > 0.0f) {
        move = f3_norm(move);
        cam.position = f3_add(cam.position,
            f3_scale(move, cam.moveSpeed * speedMul * dtSec));
    }
}

static void SyncCameraToRenderer(console::RuntimeConsole& cc,
    gfx::RendererD3D12& renderer,
    const FreeFlyCamera& cam) {
    float3 fwd, right, up;
    CamBasis(cam, fwd, right, up);
    cc.renderer.eye = cam.position;
    cc.renderer.at = f3_add(cam.position, fwd);
    cc.renderer.up = up;
    console::ApplyRendererRuntime(cc, renderer);
}

// ---------------- Win32 窗口处理 ----------------
static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_SIZE:
        if (wp != SIZE_MINIMIZED) {
            uint32_t w = LOWORD(lp), h = HIWORD(lp);
            if (auto* r = reinterpret_cast<gfx::RendererD3D12*>(
                GetWindowLongPtr(hwnd, GWLP_USERDATA)))
            {
                r->Resize(w, h);
            }
        }
        break;
    case WM_RBUTTONDOWN:
        g_RmbDown = true;
        SetCapture(hwnd);
        g_LastMouse.x = GET_X_LPARAM(lp);
        g_LastMouse.y = GET_Y_LPARAM(lp);
        ShowCursor(FALSE);
        break;
    case WM_RBUTTONUP:
        g_RmbDown = false;
        ReleaseCapture();
        ShowCursor(TRUE);
        break;
    case WM_MOUSEMOVE:
        if (g_RmbDown && g_cam) {
            POINT p{ GET_X_LPARAM(lp), GET_Y_LPARAM(lp) };
            int dx = p.x - g_LastMouse.x;
            int dy = p.y - g_LastMouse.y;
            g_LastMouse = p;
            g_cam->yaw += (double)dx * g_cam->mouseSensitivity;
            g_cam->pitch += (double)(-dy) * g_cam->mouseSensitivity;
            const double kLim = 1.5533;
            g_cam->pitch = std::clamp(g_cam->pitch, -kLim, kLim);
        }
        break;
    case WM_MOUSEWHEEL:
        g_PendingWheel += (float)GET_WHEEL_DELTA_WPARAM(wp);
        break;
    case WM_KILLFOCUS:
        if (g_RmbDown) {
            g_RmbDown = false;
            ReleaseCapture();
            ShowCursor(TRUE);
        }
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

static HWND CreateSimpleWindow(uint32_t w, uint32_t h, HINSTANCE hInst) {
    WNDCLASSW wc{};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInst;
    wc.lpszClassName = L"PBFxWnd";
    RegisterClassW(&wc);

    RECT rc{ 0,0,(LONG)w,(LONG)h };
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
    HWND hwnd = CreateWindowW(
        wc.lpszClassName, L"PBF-X",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT,
        rc.right - rc.left, rc.bottom - rc.top,
        nullptr, nullptr, hInst, nullptr);
    return hwnd;
}

// ---------------- 配置路径 ----------------
static std::string MakeConfigAbsolutePath() {
    char buf[MAX_PATH];
    DWORD len = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    std::string exePath = (len > 0) ? std::string(buf, len) : std::string();
    size_t pos = exePath.find_last_of("/\\");
    std::string dir = (pos == std::string::npos) ? std::string() : exePath.substr(0, pos);
    return dir + "/config.json";
}

// ---------------- 主入口 ----------------
int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE, PWSTR, int) {
#ifdef ENABLE_NVTX
    nvtx3::scoped_range sr_main{ "Program.Main" };
#endif

    auto& cc = console::Instance();

    // 加载配置
    config::State cfgState{};
    {
        std::string cfgPath = MakeConfigAbsolutePath();
        std::string err;
        if (!config::LoadFile(cfgPath, cc, &cfgState, &err) && !err.empty()) {
            std::fprintf(stderr, "Config load warning: %s\n", err.c_str());
        }
    }

    HWND hwnd = CreateSimpleWindow(cc.app.width, cc.app.height, hInst);

    gfx::RendererD3D12 renderer;
    SetWindowLongPtr(hwnd, GWLP_USERDATA, (LONG_PTR)&renderer);

    gfx::RenderInitParams rp;
    console::BuildRenderInitParams(cc, rp);

    if (!renderer.Initialize(hwnd, rp))
        return 1;

    // 初始化摄像机
    console::FitCameraToDomain(cc);
    console::ApplyRendererRuntime(cc, renderer);

    g_DebugEnabled = cc.debug.enabled;
    g_DebugPaused = g_DebugEnabled && cc.debug.pauseOnStart;
    g_DebugStepRequested = false;

    g_Camera = MakeCameraFromCc(cc);
    g_Camera.moveSpeed = ComputeBaseSpeedFromDomain(cc.sim.gridMins, cc.sim.gridMaxs);
    g_cam = &g_Camera;

    core::Profiler profiler;
    if (cc.sim.demoMode == console::RuntimeConsole::Simulation::DemoMode::CubeMix) {
        console::PrepareCubeMix(cc);
    }

    // 初始化模拟器
    sim::Simulator simulator;
    sim::SimParams simParams{};
    console::BuildSimParams(cc, simParams);

    if (!simulator.initialize(simParams)) {
        MessageBoxW(hwnd, L"Simulator initialize failed.", L"PBF-X", MB_ICONERROR);
        return 1;
    }

    prof::SetNvtxEnabled(cc.perf.enable_nvtx);
    prof::Mark("App.Startup", prof::Color(0xFF, 0x80, 0x20));

    // ---------------- 外部双缓冲位置共享（Ping-Pong） ----------------
    {
        const uint32_t capacity =
            (simParams.maxParticles > 0) ? simParams.maxParticles : simParams.numParticles;

        HANDLE sharedA = nullptr, sharedB = nullptr;
        renderer.CreateSharedParticleBufferIndexed(0, capacity, sizeof(float4), sharedA);
        renderer.CreateSharedParticleBufferIndexed(1, capacity, sizeof(float4), sharedB);

        size_t bytes = size_t(capacity) * sizeof(float4);
        if (!simulator.enableExternalPingPong(sharedA, bytes, sharedB, bytes)) {
            std::fprintf(stderr, "[App][Error] enableExternalPingPong failed.\n");
        }

        // Windows 侧句柄不再需要
        CloseHandle(sharedA);
        CloseHandle(sharedB);

        if (simulator.usingExternalPingPong()) {
            renderer.RegisterPingPongCudaPtrs(simulator.pingpongPosA(), simulator.pingpongPosB());
        }
        renderer.SetParticleCount(simulator.activeParticleCount());
    }

    // 绑定时间线 fence (D3D12 shared fence -> CUDA external semaphore)
    if (!simulator.bindTimelineFence(renderer.SharedTimelineFenceHandle())) {
        std::fprintf(stderr, "[App][Warn] bindTimelineFence failed, fallback to sequential usage.\n");
    }

    // ---------------- 播种粒子 ----------------
    if (cc.sim.demoMode == console::RuntimeConsole::Simulation::DemoMode::CubeMix) {
        std::vector<float3> centers;
        console::GenerateCubeMixCenters(cc, centers);

        const uint32_t groups = cc.sim.cube_group_count;
        const uint32_t edge = cc.sim.cube_edge_particles;
        const float spacing = simParams.kernel.h *
            ((cc.sim.cube_lattice_spacing_factor_h > 0.f) ? cc.sim.cube_lattice_spacing_factor_h : 1.0f);

        simulator.seedCubeMix(groups, centers.data(), edge, spacing,
            (cc.sim.cube_apply_initial_jitter && cc.sim.initial_jitter_enable),
            cc.sim.initial_jitter_scale_h * simParams.kernel.h,
            cc.sim.initial_jitter_seed);

        renderer.UpdateGroupPalette(&cc.sim.cube_group_colors[0][0], groups);
        renderer.SetParticleGrouping(groups, cc.sim.cube_particles_per_group);
    }
    else {
        const float spacing = simParams.kernel.h *
            ((cc.sim.lattice_spacing_factor_h > 0.f) ? cc.sim.lattice_spacing_factor_h : 1.0f);
        const float3 origin = make_float3(
            simParams.grid.mins.x + 0.95f * spacing,
            simParams.grid.mins.y + 0.5f * spacing,
            simParams.grid.mins.z + 0.5f * spacing);
        simulator.seedBoxLatticeAuto(simParams.numParticles, origin, spacing);
        renderer.UpdateGroupPalette(nullptr, 0);
        renderer.SetParticleGrouping(0, 0);
    }

    simParams.numParticles = simulator.activeParticleCount();
    renderer.SetParticleCount(simParams.numParticles);

    // ---------------- 主循环 ----------------
    uint64_t frameIndex = 0;
    MSG msg{};
    bool running = true;

    const uint32_t hotReloadEveryNFrames =
        (cc.app.hot_reload && cc.app.hot_reload_every_n > 0)
        ? (uint32_t)cc.app.hot_reload_every_n : UINT32_MAX;

    const uint32_t diagEveryNFrames =
        (cc.debug.diag_every_n > 0)
        ? (uint32_t)cc.debug.diag_every_n : UINT32_MAX;

    using Clock = std::chrono::steady_clock;
    static auto s_prevFrameEnd = Clock::now();

    const bool   benchEnabled = cc.bench.enabled;
    const int    benchTotalFrames = cc.bench.total_frames;
    const double benchTotalSeconds = cc.bench.total_seconds;
    const bool   benchUseTimeRange = cc.bench.use_time_range;
    const int    sampleBeginFrame = cc.bench.sample_begin_frame;
    const int    sampleEndFrameCfg = cc.bench.sample_end_frame;
    const double sampleBeginSec = cc.bench.sample_begin_seconds;
    const double sampleEndSecCfg = cc.bench.sample_end_seconds;

    auto wallStart = Clock::now();
    uint64_t benchSampledFrames = 0;
    double accumFrameMs = 0.0, accumSimGpuMs = 0.0, accumRenderMs = 0.0;
    double frameMsMin = 1e300, frameMsMax = 0.0;
    double simMsMin = 1e300, simMsMax = 0.0;
    double renderMsMin = 1e300, renderMsMax = 0.0;

    auto InFrameSampleWindow = [&](uint64_t f)->bool {
        if (benchUseTimeRange) return false;
        if ((int64_t)f < sampleBeginFrame) return false;
        if (sampleEndFrameCfg >= 0 && (int64_t)f > sampleEndFrameCfg) return false;
        return true;
        };
    auto InTimeSampleWindow = [&](double e)->bool {
        if (!benchUseTimeRange) return false;
        if (e < sampleBeginSec) return false;
        if (sampleEndSecCfg > 0.0 && e > sampleEndSecCfg) return false;
        return true;
        };

    while (running) {
        // 消息泵
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) running = false;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        if (!running) break;

        auto frameStart = Clock::now();
        double elapsedSec = std::chrono::duration<double>(frameStart - wallStart).count();

        if (benchEnabled) {
            bool fDone = (benchTotalFrames > 0) && ((int64_t)frameIndex >= benchTotalFrames);
            bool tDone = (benchTotalSeconds > 0.0) && (elapsedSec >= benchTotalSeconds);
            if (fDone || tDone) running = false;
        }
        if (!running) break;

        // 热重载
        if (hotReloadEveryNFrames != UINT32_MAX && (frameIndex % hotReloadEveryNFrames) == 0) {
            std::string err;
            if (config::TryHotReload(cfgState, cc, &err)) {
                console::BuildSimParams(cc, simParams);
                console::FitCameraToDomain(cc);
                console::ApplyRendererRuntime(cc, renderer);
                if (cc.debug.printHotReload) {
                    std::printf("[HotReload] Applied profile=%s K=%d maxN=%d sortN=%d point_size=%.2f\n",
                        cfgState.activeProfile.c_str(),
                        simParams.solverIters,
                        simParams.maxNeighbors,
                        simParams.sortEveryN,
                        cc.viewer.point_size_px);
                }
            }
        }

        // 采样/统计
        profiler.beginFrame(frameIndex);
        profiler.addCounter("num_particles", (int64_t)simParams.numParticles);
        profiler.addCounter("solver_iters", simParams.solverIters);
        profiler.addCounter("max_neighbors", simParams.maxNeighbors);
        profiler.addCounter("sort_every_n", simParams.sortEveryN);
        profiler.addText("profile", cfgState.activeProfile);

        if (diagEveryNFrames != UINT32_MAX && (frameIndex % diagEveryNFrames) == 0) {
            // 可扩展诊断占位
        }

        float dtSec = std::chrono::duration<float>(frameStart - s_prevFrameEnd).count();
        dtSec = std::clamp(dtSec, 0.0f, 0.2f);

        UpdateFreeFlyCamera(g_Camera, dtSec);
        SyncCameraToRenderer(cc, renderer, g_Camera);

        // Debug 控制
        bool doStep = true;
        if (g_DebugEnabled && !benchEnabled) {
            if (KeyPressedOnce(cc.debug.keyTogglePause)) {
                g_DebugPaused = !g_DebugPaused;
                if (cc.debug.printDebugKeys)
                    std::printf("[Debug] %s\n", g_DebugPaused ? "Paused" : "Running");
            }
            if (KeyPressedOnce(cc.debug.keyRun)) {
                g_DebugPaused = false;
                g_DebugStepRequested = false;
                if (cc.debug.printDebugKeys)
                    std::printf("[Debug] Run (continuous)\n");
            }
            if (KeyPressedOnce(cc.debug.keyStep)) {
                if (g_DebugPaused) {
                    g_DebugStepRequested = true;
                    if (cc.debug.printDebugKeys)
                        std::printf("[Debug] Step requested\n");
                }
            }
            doStep = !g_DebugPaused || g_DebugStepRequested;
        }

        // 模拟步进
        if (doStep) {
            simulator.step(simParams);
            renderer.UpdateParticleSRVForPingPong(simulator.devicePositions());
            if (g_DebugEnabled && g_DebugStepRequested) {
                g_DebugStepRequested = false;
                g_DebugPaused = true;
            }
            if (simulator.swappedThisFrame()) {
                renderer.UpdateParticleSRVForPingPong(simulator.pingpongPosB());
            }
        }

        const double simMsGpu = simulator.lastGpuFrameMs();
        profiler.addRow("sim_ms_gpu", simMsGpu);
        renderer.SetParticleCount(simulator.activeParticleCount());

        // 同步：等待模拟完成的时间线值
        renderer.WaitSimulationFence(simulator.lastSimFenceValue());

        auto renderStart = Clock::now();
        renderer.RenderFrame(profiler);
        renderer.WaitForGPU();
        auto renderEnd = Clock::now();

        const double renderMs = std::chrono::duration<double, std::milli>(renderEnd - renderStart).count();
        profiler.addRow("render_ms", renderMs);

        auto frameEnd = Clock::now();
        s_prevFrameEnd = frameEnd;
        const double frameMs = std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();
        const double fpsInst = (frameMs > 1e-6) ? (1000.0 / frameMs) : 0.0;

        static double s_fpsEma = 0.0;
        s_fpsEma = (s_fpsEma <= 0.0) ? fpsInst : (0.9 * s_fpsEma + 0.1 * fpsInst);

        profiler.addRow("frame_ms", frameMs);
        profiler.addRow("fps", fpsInst);

        UpdateWindowTitleHUD(hwnd, simParams.numParticles, s_fpsEma);

        // 基准采样
        if (benchEnabled) {
            bool inWindow = benchUseTimeRange
                ? InTimeSampleWindow(elapsedSec)
                : InFrameSampleWindow(frameIndex);
            if (inWindow) {
                ++benchSampledFrames;
                accumFrameMs += frameMs;
                accumSimGpuMs += simMsGpu;
                accumRenderMs += renderMs;
                frameMsMin = std::min(frameMsMin, frameMs);
                frameMsMax = std::max(frameMsMax, frameMs);
                simMsMin = std::min(simMsMin, simMsGpu);
                simMsMax = std::max(simMsMax, simMsGpu);
                renderMsMin = std::min(renderMsMin, renderMs);
                renderMsMax = std::max(renderMsMax, renderMs);
                if (cc.bench.print_per_frame) {
                    std::printf("[BenchFrame] f=%llu frame_ms=%.3f sim_ms=%.3f render_ms=%.3f\n",
                        (unsigned long long)frameIndex, frameMs, simMsGpu, renderMs);
                }
            }
        }

        profiler.flushCsv(cc.app.csv_path, frameIndex);
        ++frameIndex;
    }

    // 基准结果输出
    if (benchEnabled) {
        double benchElapsedSec = std::chrono::duration<double>(Clock::now() - wallStart).count();
        if (benchSampledFrames > 0) {
            double avgFrame = accumFrameMs / benchSampledFrames;
            double avgSim = accumSimGpuMs / benchSampledFrames;
            double avgRender = accumRenderMs / benchSampledFrames;
            double fpsAvg = 1000.0 / avgFrame;

            std::printf("\n==== Benchmark Result ====\n");
            std::printf("SampledFrames: %llu\n", (unsigned long long)benchSampledFrames);
            std::printf("TotalElapsedSec: %.3f\n", benchElapsedSec);
            if (!benchUseTimeRange) {
                std::printf("FrameRange: [%d, %s]\n",
                    sampleBeginFrame,
                    (cc.bench.sample_end_frame >= 0)
                    ? std::to_string(cc.bench.sample_end_frame).c_str()
                    : "end");
            }
            else {
                std::printf("TimeRangeSec: [%.3f, %s]\n",
                    sampleBeginSec,
                    (cc.bench.sample_end_seconds > 0.0)
                    ? std::to_string(cc.bench.sample_end_seconds).c_str()
                    : "end");
            }
            std::printf("Frame_ms avg=%.3f min=%.3f max=%.3f\n", avgFrame, frameMsMin, frameMsMax);
            std::printf("Sim_msGpu avg=%.3f min=%.3f max=%.3f\n", avgSim, simMsMin, simMsMax);
            std::printf("Render_ms avg=%.3f min=%.3f max=%.3f\n", avgRender, renderMsMin, renderMsMax);
            std::printf("FPS_avg=%.2f\n", fpsAvg);
            std::printf("==========================\n");
        }
        else {
            std::printf("[Benchmark] No frames sampled (check window configuration).\n");
        }
    }

    simulator.shutdown();
    renderer.Shutdown();
    return 0;
}

int main() {
    return wWinMain(GetModuleHandleW(nullptr), nullptr, GetCommandLineW(), SW_SHOWNORMAL);
}