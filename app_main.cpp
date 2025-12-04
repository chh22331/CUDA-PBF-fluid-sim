#include <cstdio>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <windows.h>
#include <windowsx.h>
#include <chrono>
#include <cmath>
#include <string> 
#include <algorithm>
#include <cstring>

#include "engine/core/config.h"
#include "engine/core/profiler.h"
#include "engine/gfx/renderer.h"
#include "engine/core/console.h"
#include "engine/core/prof_nvtx.h"
#include "sim/simulator.h"
#include "sim/parameters.h"

// =====================================================================
// Debug & Input State
// =====================================================================

static bool g_DebugEnabled = false;
static bool g_DebugPaused = false;
static bool g_DebugStepRequested = false;

// Poll virtual key state via GetAsyncKeyState (Win32 API).
static inline bool KeyDown(int vk) {
    return (GetAsyncKeyState(vk) & 0x8000) != 0;
}

// Rising-edge detection with debouncing to detect single key presses.
static bool KeyPressedOnce(int vk) {
    static uint8_t prev[512] = {};
    vk &= 0x1FF;
    bool down = KeyDown(vk);
    bool fired = down && (prev[vk] == 0);
    prev[vk] = down ? 1 : 0;
    return fired;
}

// =====================================================================
// Window Title HUD
// =====================================================================

// Update window title to show particle count and instantaneous FPS.
static void UpdateWindowTitleHUD(
    HWND hwnd,
    const console::RuntimeConsole& cc,
    uint32_t particleCount,
    double fps
) {
    wchar_t buf[256];
    if (fps < 1e-3) fps = 0.0;
    swprintf_s(buf, sizeof(buf) / sizeof(buf[0]), L"PBF-X | N=%u | FPS=%.1f", particleCount, fps);
    SetWindowTextW(hwnd, buf);
}

// =====================================================================
// Lightweight float3 Helpers
// =====================================================================

static inline float3 f3_make(float x, float y, float z) {
    return make_float3(x, y, z);
}

static inline float3 f3_add(const float3& a, const float3& b) {
    return f3_make(a.x + b.x, a.y + b.y, a.z + b.z);
}

static inline float3 f3_sub(const float3& a, const float3& b) {
    return f3_make(a.x - b.x, a.y - b.y, a.z - b.z);
}

static inline float3 f3_scale(const float3& a, float s) {
    return f3_make(a.x * s, a.y * s, a.z * s);
}

static inline float f3_dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline float3 f3_cross(const float3& a, const float3& b) {
    return f3_make(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

static inline float f3_len(const float3& a) {
    return std::sqrt(f3_dot(a, a));
}

static inline float3 f3_norm(const float3& a) {
    float l = f3_len(a);
    return (l > 1e-20f) ? f3_scale(a, 1.0f / l) : f3_make(0, 0, 0);
}

// =====================================================================
// Free-Fly Camera
// =====================================================================

struct FreeFlyCamera {
    float3 position{};
    double yaw = 0.0;    // Rotation around world-space Y axis (radians).
    double pitch = 0.0;  // Rotation around camera-space X axis (radians).
    float moveSpeed = 1.0f;         // Baseline move speed in meters/second.
    float mouseSensitivity = 0.002f; // Radians per pixel of mouse motion.
};

static FreeFlyCamera g_Camera;
static FreeFlyCamera* g_cam = nullptr;

// Mouse state for right-mouse-button drag.
static bool g_RmbDown = false;
static POINT g_LastMouse{};
static float g_PendingWheel = 0.0f;

// Compute camera forward vector from yaw/pitch.
static inline float3 CamForward(const FreeFlyCamera& c) {
    float cp = std::cos((float)c.pitch);
    float sp = std::sin((float)c.pitch);
    float cy = std::cos((float)c.yaw);
    float sy = std::sin((float)c.yaw);
    return f3_norm(f3_make(sy * cp, sp, cy * cp));
}

// Compute camera basis vectors (forward, right, up).
static inline void CamBasis(
    const FreeFlyCamera& c,
    float3& fwd,
    float3& right,
    float3& up
) {
    const float3 worldUp = f3_make(0.f, 1.f, 0.f);
    fwd = CamForward(c);
    right = f3_norm(f3_cross(fwd, worldUp));
    up = f3_cross(right, fwd);
}

// Initialize camera from console configuration to preserve initial framing.
static FreeFlyCamera MakeCameraFromCc(const console::RuntimeConsole& cc) {
    FreeFlyCamera cam{};
    cam.position = cc.renderer.eye;
    float3 fwd = f3_norm(f3_sub(cc.renderer.at, cc.renderer.eye));
    cam.pitch = std::asin(std::clamp((double)fwd.y, -1.0, 1.0));
    cam.yaw = std::atan2((double)fwd.x, (double)fwd.z);
    cam.moveSpeed = 1.0f;
    return cam;
}

// Derive baseline camera speed from domain extent (~30% of diagonal).
static float ComputeBaseSpeedFromDomain(const float3& mins, const float3& maxs) {
    float3 ext = f3_sub(maxs, mins);
    float diag = f3_len(ext);
    return std::max(0.3f, diag * 0.3f);
}

// Update camera each frame using keyboard/scrollwheel input (dtSec = wall time delta).
static void UpdateFreeFlyCamera(FreeFlyCamera& cam, float dtSec) {
    float3 fwd, right, up;
    CamBasis(cam, fwd, right, up);

    // Consume accumulated scroll wheel delta for dolly in/out.
    if (std::abs(g_PendingWheel) > 1e-4f) {
        float dolly = g_PendingWheel * 0.0025f * cam.moveSpeed;
        cam.position = f3_add(cam.position, f3_scale(fwd, dolly));
        g_PendingWheel = 0.0f;
    }

    // Keyboard-driven WASD + QE translation.
    auto down = [](int vk) -> bool {
        return (GetAsyncKeyState(vk) & 0x8000) != 0;
    };

    float3 move = f3_make(0, 0, 0);
    if (down('W')) move = f3_add(move, fwd);
    if (down('S')) move = f3_sub(move, fwd);
    if (down('D')) move = f3_add(move, right);
    if (down('A')) move = f3_sub(move, right);
    if (down('E')) move = f3_add(move, up);
    if (down('Q')) move = f3_sub(move, up);

    float speedMul = 1.0f;
    if (down(VK_SHIFT)) speedMul *= 5.0f;
    if (down(VK_CONTROL)) speedMul *= 0.2f;

    if (f3_len(move) > 0.0f) {
        move = f3_norm(move);
        cam.position = f3_add(
            cam.position,
            f3_scale(move, cam.moveSpeed * speedMul * dtSec)
        );
    }
}

// Sync camera transform to the renderer each frame.
static void SyncCameraToRenderer(
    console::RuntimeConsole& cc,
    gfx::RendererD3D12& renderer,
    const FreeFlyCamera& cam
) {
    float3 fwd, right, up;
    CamBasis(cam, fwd, right, up);

    cc.renderer.eye = cam.position;
    cc.renderer.at = f3_add(cam.position, fwd);
    cc.renderer.up = up;

    console::ApplyRendererRuntime(cc, renderer);
}

// =====================================================================
// Advanced Collapse Diagnostics (High-Cost)
// =====================================================================

// Log expensive cluster metrics for debugging collapse (guarded by debug flags).
static void LogAdvancedCollapseDiagnostics(
    const console::RuntimeConsole& cc,
    sim::Simulator& simulator,
    const sim::SimParams& simParams,
    core::Profiler& profiler
) {
    const uint32_t N = simParams.numParticles;
    if (N == 0) return;

    const float4* d_pos = simulator.devicePositions();
    if (!d_pos) return;

    const uint32_t sampleStride = (cc.debug.advancedCollapseSampleStride >= 1)
        ? (uint32_t)cc.debug.advancedCollapseSampleStride
        : 16u;

    std::vector<float4> h_pos(N);
    if (cudaMemcpy(h_pos.data(), d_pos, sizeof(float4) * N, cudaMemcpyDeviceToHost) != cudaSuccess) {
        if (cc.debug.printWarnings) {
            std::fprintf(stderr, "[Diag] cudaMemcpy positions failed.\n");
        }
        return;
    }

    // Sub-sample particles for cheaper analysis.
    std::vector<float3> pts;
    pts.reserve((N + sampleStride - 1) / sampleStride);
    for (uint32_t i = 0; i < N; i += sampleStride) {
        pts.push_back(make_float3(h_pos[i].x, h_pos[i].y, h_pos[i].z));
    }

    const size_t M = pts.size();
    if (M < 2) return;

    // Compute center of mass.
    double sx = 0, sy = 0, sz = 0;
    for (auto& p : pts) {
        sx += p.x;
        sy += p.y;
        sz += p.z;
    }
    const double invM = 1.0 / double(M);
    const double comx = sx * invM;
    const double comy = sy * invM;
    const double comz = sz * invM;

    // Cluster radius metrics (RMS / max).
    double sumR2 = 0.0;
    double maxR = 0.0;
    for (auto& p : pts) {
        const double dx = p.x - comx;
        const double dy = p.y - comy;
        const double dz = p.z - comz;
        const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
        sumR2 += r * r;
        if (r > maxR) maxR = r;
    }
    const double rmsR = std::sqrt(sumR2 * invM);

    // Nearest-neighbor distance distribution.
    std::vector<float> nn(M);
    for (size_t i = 0; i < M; ++i) {
        double best2 = 1e300;
        const double xi = pts[i].x;
        const double yi = pts[i].y;
        const double zi = pts[i].z;
        for (size_t j = 0; j < M; ++j) {
            if (j == i) continue;
            const double dx = xi - pts[j].x;
            const double dy = yi - pts[j].y;
            const double dz = zi - pts[j].z;
            const double d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < best2) best2 = d2;
        }
        nn[i] = (best2 > 0.0) ? float(std::sqrt(best2)) : 0.0f;
    }

    std::sort(nn.begin(), nn.end());

    auto pct = [&](double q) -> float {
        if (nn.empty()) return 0.0f;
        double idx = q * (nn.size() - 1);
        size_t i0 = size_t(std::floor(idx));
        size_t i1 = std::min(i0 + 1, nn.size() - 1);
        double t = idx - i0;
        return float((1.0 - t) * nn[i0] + t * nn[i1]);
    };

    const float nnMin = nn.front();
    const float nnMed = pct(0.5);
    const float nnP05 = pct(0.05);
    const float nnP95 = pct(0.95);

    // Ratio relative to collapse threshold (higher = overlapping).
    const float h = simParams.kernel.h;
    const float th05 = 0.05f * h;
    const float th10 = 0.10f * h;

    size_t cnt05 = 0, cnt10 = 0;
    for (float d : nn) {
        if (d <= th05) ++cnt05;
        if (d <= th10) ++cnt10;
    }
    const double frac05 = double(cnt05) / double(M);
    const double frac10 = double(cnt10) / double(M);

    // Write CSV snapshot (integer-scaled for profiler).
    profiler.addCounter(
        "collapse_rms_radius_over_h_x1000",
        (int64_t)llround((rmsR / h) * 1000.0)
    );
    profiler.addCounter(
        "collapse_max_radius_over_h_x1000",
        (int64_t)llround((maxR / h) * 1000.0)
    );
    profiler.addCounter(
        "nn_min_over_h_x1000",
        (int64_t)llround((nnMin / h) * 1000.0)
    );
    profiler.addCounter(
        "nn_p05_over_h_x1000",
        (int64_t)llround((nnP05 / h) * 1000.0)
    );
    profiler.addCounter(
        "nn_median_over_h_x1000",
        (int64_t)llround((nnMed / h) * 1000.0)
    );
    profiler.addCounter(
        "nn_p95_over_h_x1000",
        (int64_t)llround((nnP95 / h) * 1000.0)
    );
    profiler.addCounter(
        "frac_nn_le_0p05h_permille",
        (int64_t)llround(frac05 * 1000.0)
    );
    profiler.addCounter(
        "frac_nn_le_0p10h_permille",
        (int64_t)llround(frac10 * 1000.0)
    );

    // Human-readable warnings (gated by printWarnings).
    if (cc.debug.printWarnings) {
        if ((rmsR / h) < 0.15 && frac05 > 0.30) {
            std::printf(
                "[Warn] Collapse suspected: rmsR=%.3f h, maxR=%.3f h, "
                "nn_min=%.3f h, frac(nn<=0.05h)=%.1f%%, frac(nn<=0.1h)=%.1f%%\n",
                float(rmsR / h), float(maxR / h), float(nnMin / h),
                float(frac05 * 100.0), float(frac10 * 100.0)
            );
        } else if (frac10 > 0.60) {
            std::printf(
                "[Warn] Dense cluster: frac(nn<=0.1h)=%.1f%%, "
                "median_nn=%.3f h, rmsR=%.3f h\n",
                float(frac10 * 100.0), float(nnMed / h), float(rmsR / h)
            );
        }
    }
}

// =====================================================================
// Runtime Sanitization
// =====================================================================

// Validate and clamp guard rails after hot reload to prevent invalid configs.
static void SanitizeRuntime(
    console::RuntimeConsole& cc,
    sim::SimParams& sp
) {
    const bool log = cc.debug.printSanitize;

    // Solver iteration range: [1, 64].
    if (sp.solverIters < 1) {
        sp.solverIters = 1;
        if (log) std::fprintf(stderr, "[Sanitize] solverIters clamped to 1\n");
    }
    if (sp.solverIters > 64) {
        sp.solverIters = 64;
        if (log) std::fprintf(stderr, "[Sanitize] solverIters clamped to 64\n");
    }

    // Neighbor cap range: [8, perf.neighbor_cap or 1024].
    if (sp.maxNeighbors < 8) {
        sp.maxNeighbors = 8;
        if (log) std::fprintf(stderr, "[Sanitize] maxNeighbors clamped to 8\n");
    }
    if (cc.perf.neighbor_cap > 0 && sp.maxNeighbors > cc.perf.neighbor_cap) {
        sp.maxNeighbors = cc.perf.neighbor_cap;
        if (log) {
            std::fprintf(
                stderr,
                "[Sanitize] maxNeighbors clamped to perf.neighbor_cap=%d\n",
                cc.perf.neighbor_cap
            );
        }
    }
    if (sp.maxNeighbors > 1024) {
        sp.maxNeighbors = 1024;
        if (log) std::fprintf(stderr, "[Sanitize] maxNeighbors clamped to 1024\n");
    }

    // Enforce sort frequency >= 1.
    if (sp.sortEveryN < 1) {
        sp.sortEveryN = 1;
        if (log) std::fprintf(stderr, "[Sanitize] sortEveryN clamped to 1\n");
    }

    // Clamp viewer point size to [0.5, 32.0] pixels.
    if (cc.viewer.point_size_px < 0.5f) {
        cc.viewer.point_size_px = 0.5f;
        if (log) {
            std::fprintf(stderr, "[Sanitize] viewer.point_size_px clamped to 0.5\n");
        }
    }
    if (cc.viewer.point_size_px > 32.f) {
        cc.viewer.point_size_px = 32.f;
        if (log) {
            std::fprintf(stderr, "[Sanitize] viewer.point_size_px clamped to 32\n");
        }
    }
}

// =====================================================================
// Config Path Utilities
// =====================================================================

// Build absolute path to config.json relative to the executable.
static std::string MakeConfigAbsolutePath() {
    wchar_t buf[MAX_PATH] = {};
    DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    std::wstring exePath(buf, (n ? n : 0));
    size_t slash = exePath.find_last_of(L"\\/");
    std::wstring dir = (slash == std::wstring::npos) ? L"." : exePath.substr(0, slash);
    std::wstring wfull = dir + L"\\config.json";

    // Convert wide string to UTF-8.
    int len = WideCharToMultiByte(
        CP_UTF8, 0, wfull.c_str(), (int)wfull.size(),
        nullptr, 0, nullptr, nullptr
    );
    std::string out(len, '\0');
    if (len > 0) {
        WideCharToMultiByte(
            CP_UTF8, 0, wfull.c_str(), (int)wfull.size(),
            &out[0], len, nullptr, nullptr
        );
    }
    return out;
}

// =====================================================================
// Win32 Window Procedure
// =====================================================================

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_SIZE:
        if (wp != SIZE_MINIMIZED) {
            uint32_t w = LOWORD(lp);
            uint32_t h = HIWORD(lp);
            gfx::RendererD3D12* r = reinterpret_cast<gfx::RendererD3D12*>(
                GetWindowLongPtr(hwnd, GWLP_USERDATA)
            );
            if (r) r->Resize(w, h);
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

            const double kLim = 1.5533; // ~89 degrees.
            if (g_cam->pitch > kLim) g_cam->pitch = kLim;
            if (g_cam->pitch < -kLim) g_cam->pitch = -kLim;
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

// =====================================================================
// Window Creation
// =====================================================================

static HWND CreateSimpleWindow(uint32_t w, uint32_t h, HINSTANCE hInst) {
    WNDCLASSW wc{};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInst;
    wc.lpszClassName = L"PBFxWnd";
    RegisterClassW(&wc);

    RECT rc{ 0, 0, (LONG)w, (LONG)h };
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);

    HWND hwnd = CreateWindowW(
        wc.lpszClassName,
        L"PBF-X",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT,
        rc.right - rc.left, rc.bottom - rc.top,
        nullptr, nullptr, hInst, nullptr
    );
    return hwnd;
}

// =====================================================================
// Main Entry Point
// =====================================================================

int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE, PWSTR, int) {
#ifdef ENABLE_NVTX
    nvtx3::scoped_range sr_main{ "Program.Main" };
#endif

    // 1) Load configuration from config.json next to the executable.
    auto& cc = console::Instance();
    config::State cfgState{};
    {
        std::string cfgPath = MakeConfigAbsolutePath();
        std::string err;
        if (!config::LoadFile(cfgPath, cc, &cfgState, &err) && !err.empty()) {
            std::fprintf(stderr, "Config load warning: %s\n", err.c_str());
        }
    }

    // 2) Create the window using config-specified resolution/vsync.
    HWND hwnd = CreateSimpleWindow(cc.app.width, cc.app.height, hInst);

    // 3) Initialize the renderer.
    gfx::RendererD3D12 renderer;
    SetWindowLongPtr(hwnd, GWLP_USERDATA, (LONG_PTR)&renderer);

    gfx::RenderInitParams rp;
    console::BuildRenderInitParams(cc, rp);
    if (!renderer.Initialize(hwnd, rp)) return 1;

    console::ApplyRendererRuntime(cc, renderer);

    // Debug mode: pause on startup if configured.
    g_DebugEnabled = cc.debug.enabled;
    g_DebugPaused = cc.debug.pauseOnStart && g_DebugEnabled;
    g_DebugStepRequested = false;

    // Initialize free-fly camera from console configuration.
    g_Camera = MakeCameraFromCc(cc);
    g_cam = &g_Camera;

    core::Profiler profiler;

    // CubeMix demo mode: partition particle sets, fit bounds, assign colors.
    if (cc.sim.demoMode == console::RuntimeConsole::Simulation::DemoMode::CubeMix) {
        console::PrepareCubeMix(cc);
    }

    // 4) Initialize the simulator.
    sim::Simulator simulator;
    sim::SimParams simParams{};
    console::BuildSimParams(cc, simParams);
    SanitizeRuntime(cc, simParams);

    g_Camera.moveSpeed = ComputeBaseSpeedFromDomain(
        simParams.grid.mins,
        simParams.grid.maxs
    );

    if (!simulator.initialize(simParams)) {
        MessageBoxW(hwnd, L"Simulator initialize failed.", L"PBF-X", MB_ICONERROR);
        return 1;
    }

    // Configure NVTX markers based on runtime flags.
    prof::SetNvtxEnabled(cc.perf.enable_nvtx);
    prof::Mark("App.Startup", prof::Color(0xFF, 0x80, 0x20));

    const uint32_t capacity = (simParams.maxParticles > 0)
        ? simParams.maxParticles
        : simParams.numParticles;

    // 5) Bind shared CUDA/D3D12 ping-pong position buffers.
    HANDLE sharedA = nullptr, sharedB = nullptr;
    renderer.CreateSharedParticleBufferIndexed(0, capacity, sizeof(float4), sharedA);
    renderer.CreateSharedParticleBufferIndexed(1, capacity, sizeof(float4), sharedB);

    size_t bytes = size_t(capacity) * sizeof(float4);
    if (!simulator.bindExternalPosPingPong(sharedA, bytes, sharedB, bytes)) {
        std::fprintf(stderr, "[App][Error] bindExternalPosPingPong failed.\n");
    }
    CloseHandle(sharedA);
    CloseHandle(sharedB);

    // Bind shared velocity buffer.
    HANDLE sharedVel = nullptr;
    if (renderer.CreateSharedVelocityBuffer(capacity, sizeof(float4), sharedVel)) {
        if (!simulator.bindExternalVelocityBuffer(sharedVel, bytes, sizeof(float4))) {
            std::fprintf(stderr, "[App][Warn] bindExternalVelocityBuffer failed.\n");
        }
    } else {
        std::fprintf(stderr, "[App][Warn] CreateSharedVelocityBuffer failed.\n");
    }
    if (sharedVel) CloseHandle(sharedVel);

    renderer.RegisterPingPongCudaPtrs(
        simulator.pingpongPosA(),
        simulator.pingpongPosB()
    );
    renderer.SetParticleCount(simulator.activeParticleCount());

    // Bind timeline fence for zero-CPU-polling synchronization.
    if (!simulator.bindTimelineFence(renderer.SharedTimelineFenceHandle())) {
        std::fprintf(
            stderr,
            "[App][Warn] bindTimelineFence failed, fallback to sequential usage.\n"
        );
    }

    // 6) Seed initial particles using CubeMix lattice.
    {
        std::vector<float3> centers;
        console::GenerateCubeMixCenters(cc, centers);

        const uint32_t groups = cc.sim.cube_group_count;
        const uint32_t edge = cc.sim.cube_edge_particles;
        const float spacing = simParams.kernel.h *
            ((cc.sim.cube_lattice_spacing_factor_h > 0.f)
                ? cc.sim.cube_lattice_spacing_factor_h
                : 1.0f);

        simulator.seedCubeMix(
            groups,
            centers.data(),
            edge,
            spacing,
            (cc.sim.cube_apply_initial_jitter && cc.sim.initial_jitter_enable),
            cc.sim.initial_jitter_scale_h * simParams.kernel.h,
            cc.sim.initial_jitter_seed
        );

        renderer.UpdateGroupPalette(&cc.sim.cube_group_colors[0][0], groups);
        renderer.SetParticleGrouping(groups, cc.sim.cube_particles_per_group);
    }

    const uint32_t active = simulator.activeParticleCount();
    simParams.numParticles = active;
    renderer.SetParticleCount(active);

    // Main loop state.
    uint64_t frameIndex = 0;
    MSG msg{};
    bool running = true;

    // Hot reload interval (in frames).
    const uint32_t hotReloadEveryNFrames =
        (cc.app.hot_reload && cc.app.hot_reload_every_n > 0)
        ? (uint32_t)cc.app.hot_reload_every_n
        : UINT32_MAX;

    // Frame timing for camera updates.
    using Clock = std::chrono::steady_clock;
    static auto s_prevFrameEnd = Clock::now();

    // 7) Benchmark configuration.
    const bool benchEnabled = cc.bench.enabled;
    const int benchTotalFrames = cc.bench.total_frames;
    const double benchTotalSeconds = cc.bench.total_seconds;
    const bool benchUseTimeRange = cc.bench.use_time_range;
    const int sampleBeginFrame = cc.bench.sample_begin_frame;
    const int sampleEndFrameCfg = cc.bench.sample_end_frame;
    const double sampleBeginSec = cc.bench.sample_begin_seconds;
    const double sampleEndSecCfg = cc.bench.sample_end_seconds;

    auto wallStart = Clock::now();

    // Aggregated stats for the measurement window.
    uint64_t benchSampledFrames = 0;
    double accumFrameMs = 0.0;
    double accumSimGpuMs = 0.0;
    double accumRenderMs = 0.0;
    double frameMsMin = 1e300, frameMsMax = 0.0;
    double simMsMin = 1e300, simMsMax = 0.0;
    double renderMsMin = 1e300, renderMsMax = 0.0;

    auto InFrameSampleWindow = [&](uint64_t frameIdx) -> bool {
        if (benchUseTimeRange) return false;
        if ((int64_t)frameIdx < sampleBeginFrame) return false;
        if (sampleEndFrameCfg >= 0 && (int64_t)frameIdx > sampleEndFrameCfg) return false;
        return true;
    };

    auto InTimeSampleWindow = [&](double elapsedSec) -> bool {
        if (!benchUseTimeRange) return false;
        if (elapsedSec < sampleBeginSec) return false;
        if (sampleEndSecCfg > 0.0 && elapsedSec > sampleEndSecCfg) return false;
        return true;
    };

    // =====================================================================
    // Main Loop
    // =====================================================================

    while (running) {
        // Process window messages.
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) running = false;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        auto frameStart = Clock::now();
        double elapsedSec = std::chrono::duration<double>(frameStart - wallStart).count();

        // Benchmark end-condition check.
        if (benchEnabled) {
            bool reachedFrames = (benchTotalFrames > 0) &&
                ((int64_t)frameIndex >= benchTotalFrames);
            bool reachedTime = (benchTotalSeconds > 0.0) &&
                (elapsedSec >= benchTotalSeconds);
            if (reachedFrames || reachedTime) {
                running = false;
            }
        }
        if (!running) break;

        // Hot reload configuration every N frames.
        if (hotReloadEveryNFrames != UINT32_MAX &&
            (frameIndex % hotReloadEveryNFrames) == 0)
        {
            std::string err;
            if (config::TryHotReload(cfgState, cc, &err)) {
                console::BuildSimParams(cc, simParams);
                SanitizeRuntime(cc, simParams);
                console::ApplyRendererRuntime(cc, renderer);
            }
            if (cc.debug.printHotReload) {
                std::printf(
                    "[HotReload] Applied profile=%s K=%d maxN=%d sortN=%d point_size=%.2f\n",
                    cfgState.activeProfile.c_str(),
                    simParams.solverIters,
                    simParams.maxNeighbors,
                    simParams.sortEveryN,
                    cc.viewer.point_size_px
                );
            }
        }

        profiler.beginFrame(frameIndex);

        // Telemetry counters.
        profiler.addCounter("num_particles", (int64_t)simParams.numParticles);
        profiler.addCounter("solver_iters", simParams.solverIters);
        profiler.addCounter("max_neighbors", simParams.maxNeighbors);
        profiler.addCounter("sort_every_n", simParams.sortEveryN);
        profiler.addText("profile", cfgState.activeProfile);

        // Update camera with wall clock delta time.
        float dtSec = std::chrono::duration<float>(frameStart - s_prevFrameEnd).count();
        dtSec = std::clamp(dtSec, 0.0f, 0.2f);
        UpdateFreeFlyCamera(g_Camera, dtSec);
        SyncCameraToRenderer(cc, renderer, g_Camera);

        // Handle debug pause/step/run commands.
        if (g_DebugEnabled && !benchEnabled) {
            if (KeyPressedOnce(cc.debug.keyTogglePause)) {
                g_DebugPaused = !g_DebugPaused;
            }
            if (KeyPressedOnce(cc.debug.keyRun)) {
                g_DebugPaused = false;
                g_DebugStepRequested = false;
            }
            if (KeyPressedOnce(cc.debug.keyStep) && g_DebugPaused) {
                g_DebugStepRequested = true;
            }
        }

        // Determine whether to advance simulation this frame.
        bool doStep = true;
        if (g_DebugEnabled && !benchEnabled) {
            doStep = !g_DebugPaused || g_DebugStepRequested;
        }

        // Simulation step.
        if (doStep) {
            if (!simulator.step(simParams)) {
                std::fprintf(stderr, "[App][Error] simulator.step failed.\n");
            }
            renderer.UpdateParticleSRVForPingPong(simulator.renderPositionPtr());

            if (g_DebugEnabled && g_DebugStepRequested) {
                g_DebugStepRequested = false;
                g_DebugPaused = true;
            }
        }

        const double simMsGpu = simulator.lastGpuFrameMs();
        profiler.addRow("sim_ms_gpu", simMsGpu);

        // Render frame.
        renderer.SetParticleCount(simulator.activeParticleCount());
        renderer.WaitSimulationFence(simulator.lastSimFenceValue());

        auto renderStart = Clock::now();
        if (!simulator.externalPingPongEnabled()) {
            renderer.UpdateParticleSRVForPingPong(simulator.renderPositionPtr());
        }
        renderer.RenderFrame(profiler);
        renderer.WaitForGPU();
        auto renderEnd = Clock::now();

        const double renderMs = std::chrono::duration<double, std::milli>(
            renderEnd - renderStart
        ).count();
        profiler.addRow("render_ms", renderMs);

        auto frameEnd = Clock::now();
        s_prevFrameEnd = frameEnd;

        const double frameMs = std::chrono::duration<double, std::milli>(
            frameEnd - frameStart
        ).count();
        const double fpsInst = (frameMs > 1e-6) ? (1000.0 / frameMs) : 0.0;

        static double s_fpsEma = 0.0;
        s_fpsEma = (s_fpsEma <= 0.0) ? fpsInst : (0.9 * s_fpsEma + 0.1 * fpsInst);

        profiler.addRow("frame_ms", frameMs);
        profiler.addRow("fps", fpsInst);

        UpdateWindowTitleHUD(hwnd, cc, simulator.activeParticleCount(), s_fpsEma);

        // Benchmark sample collection.
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
                    std::printf(
                        "[BenchFrame] f=%llu frame_ms=%.3f sim_ms=%.3f render_ms=%.3f\n",
                        (unsigned long long)frameIndex,
                        frameMs, simMsGpu, renderMs
                    );
                }
            }
        }

        profiler.flushCsv(cc.app.csv_path, frameIndex);
        ++frameIndex;
    }

    // =====================================================================
    // Benchmark Report
    // =====================================================================

    if (benchEnabled) {
        double benchElapsedSec = std::chrono::duration<double>(
            Clock::now() - wallStart
        ).count();

        if (benchSampledFrames > 0) {
            double avgFrame = accumFrameMs / benchSampledFrames;
            double avgSim = accumSimGpuMs / benchSampledFrames;
            double avgRender = accumRenderMs / benchSampledFrames;
            double fps = 1000.0 / avgFrame;

            std::printf("\n==== Benchmark Result ====\n");
            std::printf("SampledFrames: %llu\n", (unsigned long long)benchSampledFrames);
            std::printf("TotalElapsedSec: %.3f\n", benchElapsedSec);

            if (!benchUseTimeRange) {
                std::printf(
                    "FrameRange: [%d, %s]\n",
                    sampleBeginFrame,
                    (cc.bench.sample_end_frame >= 0)
                        ? std::to_string(cc.bench.sample_end_frame).c_str()
                        : "end"
                );
            } else {
                std::printf(
                    "TimeRangeSec: [%.3f, %s]\n",
                    sampleBeginSec,
                    (cc.bench.sample_end_seconds > 0.0)
                        ? std::to_string(cc.bench.sample_end_seconds).c_str()
                        : "end"
                );
            }

            std::printf(
                "Frame_ms avg=%.3f min=%.3f max=%.3f\n",
                avgFrame, frameMsMin, frameMsMax
            );
            std::printf("FPS_avg=%.2f\n", fps);
            std::printf("==========================\n");
        } else {
            std::printf(
                "[Benchmark] No frames sampled (check window configuration).\n"
            );
        }
    }

    simulator.shutdown();
    renderer.Shutdown();
    return 0;
}

int main() {
    return wWinMain(GetModuleHandleW(nullptr), nullptr, GetCommandLineW(), SW_SHOWNORMAL);
}
