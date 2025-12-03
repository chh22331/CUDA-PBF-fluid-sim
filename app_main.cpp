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
#include "engine/audio/audio_reactive.h"
#include "sim/simulator.h"
#include "sim/parameters.h"
#include "sim/audio_forces.h"
// ---- Debug 全局状态与工具 ----
static bool g_DebugEnabled = false;
static bool g_DebugPaused = false;
static bool g_DebugStepRequested = false;
// 使用 GetAsyncKeyState 轮询；只 ?app_main.cpp 中包 ?windows.h 已满足依 ?
static inline bool KeyDown(int vk) {
    return (GetAsyncKeyState(vk) & 0x8000) != 0;
}
// 更新窗口标题栏（粒子数/FPS/音频调参）
static void UpdateWindowTitleHUD(HWND hwnd, const console::RuntimeConsole& cc, uint32_t particleCount, double fps) {
    wchar_t buf[256];
    if (fps < 1e-3) fps = 0.0;
    if (cc.audio.enabled && cc.audio.printDebug) {
        swprintf(buf, L"PBF-X | N=%u | FPS=%.1f | Gain=%.1f | Gate=%.1f dB",
            particleCount, fps, cc.audio.intensityGain, cc.audio.noiseGateDb);
    }
    else {
        swprintf(buf, L"PBF-X | N=%u | FPS=%.1f", particleCount, fps);
    }
    SetWindowTextW(hwnd, buf);
}
// 上升沿触发（去抖 ?
static bool KeyPressedOnce(int vk) {
    static uint8_t prev[512] = {};
    vk &= 0x1FF; // 简单限 ?
    bool down = KeyDown(vk);
    bool fired = down && (prev[vk] == 0);
    prev[vk] = down ? 1 : 0;
    return fired;
}
static bool AdjustWithKey(int vk, float delta, float minV, float maxV, float& value) {
    if (!KeyPressedOnce(vk)) return false;
    value = std::clamp(value + delta, minV, maxV);
    return true;
}
static void HandleAudioTuningInput(console::RuntimeConsole& cc, audio::AudioReactiveSystem& audioSystem) {
    if (!cc.audio.enabled || !cc.audio.printDebug || !audioSystem.isActive()) return;
    bool changed = false;
    changed |= AdjustWithKey(VK_F5, -1.0f, 0.0f, 200.0f, cc.audio.intensityGain);
    changed |= AdjustWithKey(VK_F6, +1.0f, 0.0f, 200.0f, cc.audio.intensityGain);
    changed |= AdjustWithKey(VK_F7, -1.0f, -120.0f, 0.0f, cc.audio.noiseGateDb);
    changed |= AdjustWithKey(VK_F8, +1.0f, -120.0f, 0.0f, cc.audio.noiseGateDb);
    if (changed) {
        audioSystem.updateGainAndGate(cc.audio.intensityGain, cc.audio.noiseGateDb);
        std::printf("[Audio][UI] intensityGain=%.1f noiseGate=%.1f dB\n",
            cc.audio.intensityGain, cc.audio.noiseGateDb);
    }
}
struct PoolDemoDebugSweepState {
    uint32_t keyCursor = 0;
    float    timer = 0.0f;
    uint32_t frameSeed = 0;
};
static PoolDemoDebugSweepState g_PoolDemoDebugState;
static sim::AudioFrameData     g_PoolDemoDebugFrame;
static void ResetPoolDemoDebugState() {
    g_PoolDemoDebugState = {};
    g_PoolDemoDebugFrame = {};
}
static const sim::AudioFrameData& NextPoolDemoDebugFrame(uint32_t requestedKeys, float dtSeconds) {
    uint32_t keyCount = (requestedKeys > 0) ? requestedKeys : sim::kAudioKeyCount;
    keyCount = std::min<uint32_t>(keyCount, sim::kAudioKeyCount);
    if (keyCount == 0u) keyCount = 1u;
    float dt = (dtSeconds > 0.0f) ? dtSeconds : 0.016f;
    auto& state = g_PoolDemoDebugState;
    const float holdSeconds = 0.2f;
    state.timer += dt;
    if (state.timer >= holdSeconds) {
        state.timer -= holdSeconds;
        state.keyCursor = (state.keyCursor + 1) % keyCount;
    }
    std::fill(std::begin(g_PoolDemoDebugFrame.keyIntensities),
        std::end(g_PoolDemoDebugFrame.keyIntensities), 0.0f);
    g_PoolDemoDebugFrame.keyIntensities[state.keyCursor] = 1.0f;
    g_PoolDemoDebugFrame.globalEnergy = 1.0f;
    g_PoolDemoDebugFrame.globalEnergyEma = 1.0f;
    g_PoolDemoDebugFrame.isBeat = 1;
    g_PoolDemoDebugFrame.beatStrength = 1.0f;
    g_PoolDemoDebugFrame.frameSeed = ++state.frameSeed;
    return g_PoolDemoDebugFrame;
}
// 简 ?float3 工具
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
    float l = f3_len(a); return (l > 1e-20f) ? f3_scale(a, 1.0f / l) : f3_make(0, 0, 0);
}
// 自由飞行相机
struct FreeFlyCamera {
    float3 position{};
    double yaw = 0.0;    // 绕世 ?Y
    double pitch = 0.0;  // 绕相 ?X
    float  moveSpeed = 1.0f;        // 基础移动速度（单 ?秒）
    float  mouseSensitivity = 0.002f; // 弧度/像素
};
static FreeFlyCamera g_Camera;
static FreeFlyCamera* g_cam = nullptr;
static bool  g_RmbDown = false;
static POINT g_LastMouse{};
static float g_PendingWheel = 0.0f;
// yaw/pitch -> 前向/ ? ?
static inline float3 CamForward(const FreeFlyCamera& c) {
    float cp = std::cos((float)c.pitch), sp = std::sin((float)c.pitch);
    float cy = std::cos((float)c.yaw), sy = std::sin((float)c.yaw);
    // yaw=0 指向 +Z；pitch 正向 ?
    return f3_norm(f3_make(sy * cp, sp, cy * cp));
}
static inline void CamBasis(const FreeFlyCamera& c, float3& fwd, float3& right, float3& up) {
    const float3 worldUp = f3_make(0.f, 1.f, 0.f);
    fwd = CamForward(c);
    right = f3_norm(f3_cross(fwd, worldUp));
    up = f3_cross(right, fwd);
}
//  ?cc 初始化相机（保持当前视角 ?
static FreeFlyCamera MakeCameraFromCc(const console::RuntimeConsole& cc) {
    FreeFlyCamera cam{};
    cam.position = cc.renderer.eye;
    float3 fwd = f3_norm(f3_sub(cc.renderer.at, cc.renderer.eye));
    // 反解 yaw/pitch
    cam.pitch = std::asin(std::clamp((double)fwd.y, -1.0, 1.0));
    cam.yaw = std::atan2((double)fwd.x, (double)fwd.z);
    cam.moveSpeed = 1.0f; // 启动后根据域尺寸再调 ?
    return cam;
}
// 根据域尺度设定基础速度（对 10x10x10 域给 ~3.0 ?
static float ComputeBaseSpeedFromDomain(const float3& mins, const float3& maxs) {
    float3 ext = f3_sub(maxs, mins);
    float  diag = f3_len(ext);
    return (((0.3f) > (diag * 0.3f)) ? (0.3f) : (diag * 0.3f));
}
// 每帧更新（键 ?滚轮），dtSec 用实时间 ?
static void UpdateFreeFlyCamera(FreeFlyCamera& cam, float dtSec) {
    float3 fwd, right, up;
    CamBasis(cam, fwd, right, up);
    // 累积的滚轮推 ?
    if (std::abs(g_PendingWheel) > 1e-4f) {
        float dolly = g_PendingWheel * 0.0025f * cam.moveSpeed; // 经验系数
        cam.position = f3_add(cam.position, f3_scale(fwd, dolly));
        g_PendingWheel = 0.0f;
    }
    // 键盘移动
    auto down = [](int vk) -> bool { return (GetAsyncKeyState(vk) & 0x8000) != 0; };
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
        cam.position = f3_add(cam.position, f3_scale(move, cam.moveSpeed * speedMul * dtSec));
    }
}
// 将相机同步到渲染器（每帧调用 ?
static void SyncCameraToRenderer(console::RuntimeConsole& cc, gfx::RendererD3D12& renderer, const FreeFlyCamera& cam) {
    float3 fwd, right, up;
    CamBasis(cam, fwd, right, up);
    cc.renderer.eye = cam.position;
    cc.renderer.at = f3_add(cam.position, fwd);
    cc.renderer.up = up;
    console::ApplyRendererRuntime(cc, renderer);
}
static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_SIZE:
        if (wp != SIZE_MINIMIZED) {
            uint32_t w = LOWORD(lp), h = HIWORD(lp);
            gfx::RendererD3D12* r = reinterpret_cast<gfx::RendererD3D12*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
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
            g_cam->pitch += (double)(-dy) * g_cam->mouseSensitivity; // 鼠标上移 -> 抬头
            const double kLim = 1.5533; // ~89 ?
            if (g_cam->pitch > kLim) g_cam->pitch = kLim;
            if (g_cam->pitch < -kLim) g_cam->pitch = -kLim;
        }
        break;
    case WM_MOUSEWHEEL:
        g_PendingWheel += (float)GET_WHEEL_DELTA_WPARAM(wp); // 累积，稍后在帧更新中消费
        break;
    case WM_KILLFOCUS:
        if (g_RmbDown) {
            g_RmbDown = false;
            ReleaseCapture();
            ShowCursor(TRUE);
        }
        break;
    case WM_DESTROY:
        PostQuitMessage(0); return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}
static HWND CreateSimpleWindow(uint32_t w, uint32_t h, HINSTANCE hInst) {
    WNDCLASSW wc{}; wc.lpfnWndProc = WndProc; wc.hInstance = hInst; wc.lpszClassName = L"PBFxWnd";
    RegisterClassW(&wc);
    RECT rc{ 0,0,(LONG)w,(LONG)h }; AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
    HWND hwnd = CreateWindowW(wc.lpszClassName, L"PBF-X", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT, rc.right - rc.left, rc.bottom - rc.top, nullptr, nullptr, hInst, nullptr);
    return hwnd;
}
// 高开销“塌陷诊断”：打印 ?cc.debug.printWarnings 控制，执行受 cc.debug.enableAdvancedCollapseDiag 控制
static void LogAdvancedCollapseDiagnostics(const console::RuntimeConsole& cc, sim::Simulator& simulator, const sim::SimParams& simParams, core::Profiler& profiler) {
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
    // 子采 ?
    std::vector<float3> pts;
    pts.reserve((N + sampleStride - 1) / sampleStride);
    for (uint32_t i = 0; i < N; i += sampleStride) {
        pts.push_back(make_float3(h_pos[i].x, h_pos[i].y, h_pos[i].z));
    }
    const size_t M = pts.size();
    if (M < 2) return;
    // 质心
    double sx = 0, sy = 0, sz = 0;
    for (auto& p : pts) { sx += p.x; sy += p.y; sz += p.z; }
    const double invM = 1.0 / double(M);
    const double comx = sx * invM, comy = sy * invM, comz = sz * invM;
    // 团簇半径（RMS / Max ?
    double sumR2 = 0.0; double maxR = 0.0;
    for (auto& p : pts) {
        const double dx = p.x - comx, dy = p.y - comy, dz = p.z - comz;
        const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
        sumR2 += r * r;
        if (r > maxR) maxR = r;
    }
    const double rmsR = std::sqrt(sumR2 * invM);
    // 最近邻距离
    std::vector<float> nn;
    nn.resize(M);
    for (size_t i = 0; i < M; ++i) {
        double best2 = 1e300;
        const double xi = pts[i].x, yi = pts[i].y, zi = pts[i].z;
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
        size_t i1 = (((i0 + 1) < (nn.size() - 1)) ? (i0 + 1) : (nn.size() - 1));
        double t = idx - i0;
        return float((1.0 - t) * nn[i0] + t * nn[i1]);
        };
    const float nnMin = nn.front();
    const float nnMed = pct(0.5);
    const float nnP05 = pct(0.05);
    const float nnP95 = pct(0.95);
    // 阈值比例（越大越接近“塌 ?重叠”）
    const float h = simParams.kernel.h;
    const float th05 = 0.05f * h;
    const float th10 = 0.10f * h;
    size_t cnt05 = 0, cnt10 = 0;
    for (float d : nn) { if (d <= th05) ++cnt05; if (d <= th10) ++cnt10; }
    const double frac05 = double(cnt05) / double(M);
    const double frac10 = double(cnt10) / double(M);
    // 写入 CSV（整型缩放）
    profiler.addCounter("collapse_rms_radius_over_h_x1000", (int64_t)llround((rmsR / h) * 1000.0));
    profiler.addCounter("collapse_max_radius_over_h_x1000", (int64_t)llround((maxR / h) * 1000.0));
    profiler.addCounter("nn_min_over_h_x1000", (int64_t)llround((nnMin / h) * 1000.0));
    profiler.addCounter("nn_p05_over_h_x1000", (int64_t)llround((nnP05 / h) * 1000.0));
    profiler.addCounter("nn_median_over_h_x1000", (int64_t)llround((nnMed / h) * 1000.0));
    profiler.addCounter("nn_p95_over_h_x1000", (int64_t)llround((nnP95 / h) * 1000.0));
    profiler.addCounter("frac_nn_le_0p05h_permille", (int64_t)llround(frac05 * 1000.0));
    profiler.addCounter("frac_nn_le_0p10h_permille", (int64_t)llround(frac10 * 1000.0));
    // 人类可读的阈值报警（ ?printWarnings 控制 ?
    if (cc.debug.printWarnings) {
        if ((rmsR / h) < 0.15 && frac05 > 0.30) {
            std::printf("[Warn] Collapse suspected: rmsR=%.3f h, maxR=%.3f h, nn_min=%.3f h, frac(nn<=0.05h)=%.1f%%, frac(nn<=0.1h)=%.1f%%\n",
                float(rmsR / h), float(maxR / h), float(nnMin / h), float(frac05 * 100.0), float(frac10 * 100.0));
        }
        else if (frac10 > 0.60) {
            std::printf("[Warn] Dense cluster: frac(nn<=0.1h)=%.1f%%, median_nn=%.3f h, rmsR=%.3f h\n",
                float(frac10 * 100.0), float(nnMed / h), float(rmsR / h));
        }
    }
}
// 最小守门阈值校验与钳制（热加载后调用）
static void SanitizeRuntime(console::RuntimeConsole& cc, sim::SimParams& sp) {
    const bool log = cc.debug.printSanitize;
    // 迭代次数 ?..64
    if (sp.solverIters < 1) { sp.solverIters = 1; if (log) std::fprintf(stderr, "[Sanitize] solverIters clamped to 1\n"); }
    if (sp.solverIters > 64) { sp.solverIters = 64; if (log) std::fprintf(stderr, "[Sanitize] solverIters clamped to 64\n"); }
    // 邻域上限 ?..1024（并 ?perf.neighbor_cap 一致）
    if (sp.maxNeighbors < 8) { sp.maxNeighbors = 8; if (log) std::fprintf(stderr, "[Sanitize] maxNeighbors clamped to 8\n"); }
    if (cc.perf.neighbor_cap > 0 && sp.maxNeighbors > cc.perf.neighbor_cap) {
        sp.maxNeighbors = cc.perf.neighbor_cap;
        if (log) std::fprintf(stderr, "[Sanitize] maxNeighbors clamped to perf.neighbor_cap=%d\n", cc.perf.neighbor_cap);
    }
    if (sp.maxNeighbors > 1024) { sp.maxNeighbors = 1024; if (log) std::fprintf(stderr, "[Sanitize] maxNeighbors clamped to 1024\n"); }
    // 排序频率 ?=1
    if (sp.sortEveryN < 1) { sp.sortEveryN = 1; if (log) std::fprintf(stderr, "[Sanitize] sortEveryN clamped to 1\n"); }
    // 视图点大小：0.5..32
    if (cc.viewer.point_size_px < 0.5f) { cc.viewer.point_size_px = 0.5f; if (log) std::fprintf(stderr, "[Sanitize] viewer.point_size_px clamped to 0.5\n"); }
    if (cc.viewer.point_size_px > 32.f) { cc.viewer.point_size_px = 32.f; if (log) std::fprintf(stderr, "[Sanitize] viewer.point_size_px clamped to 32\n"); }
}
static std::string MakeConfigAbsolutePath() {
    wchar_t buf[MAX_PATH] = {};
    DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    std::wstring exePath(buf, (n ? n : 0));
    size_t slash = exePath.find_last_of(L"\\/");
    std::wstring dir = (slash == std::wstring::npos) ? L"." : exePath.substr(0, slash);
    std::wstring wfull = dir + L"\\config.json";
    //  ?UTF-8
    int len = WideCharToMultiByte(CP_UTF8, 0, wfull.c_str(), (int)wfull.size(), nullptr, 0, nullptr, nullptr);
    std::string out(len, '\0');
    if (len > 0) {
        WideCharToMultiByte(CP_UTF8, 0, wfull.c_str(), (int)wfull.size(), &out[0], len, nullptr, nullptr);
    }
    return out;
}
static audio::AudioReactiveSystem::Settings MakeAudioSettings(const console::RuntimeConsole& cc) {
    const auto& cfg = cc.audio;
    audio::AudioReactiveSystem::Settings s{};
    s.enabled = cfg.enabled;
    s.sampleRate = cfg.sampleRate;
    s.channels = cfg.channels;
    s.preferLoopback = cfg.preferLoopback || cfg.capturePlayback;
    s.fallbackToCapture = cfg.fallbackToCapture;
    s.fftSize = cfg.fftSize;
    s.ringBufferMs = cfg.ringBufferMs;
    s.keyCount = cfg.keyCount;
    s.minFrequencyHz = cfg.minFrequencyHz;
    s.maxFrequencyHz = cfg.maxFrequencyHz;
    s.intensityGain = cfg.intensityGain;
    s.noiseGateDb = cfg.noiseGateDb;
    s.smoothingAttackSec = cfg.smoothingAttackSec;
    s.smoothingReleaseSec = cfg.smoothingReleaseSec;
    s.beatThreshold = cfg.beatThreshold;
    s.beatHoldSeconds = cfg.beatHoldSeconds;
    s.beatReleaseSeconds = cfg.beatReleaseSeconds;
    s.globalEnergyEmaSeconds = cfg.globalEnergyEmaSeconds;
    s.debugPrint = cfg.printDebug;
    return s;
}
static bool AudioSettingsEqual(const audio::AudioReactiveSystem::Settings& a,
    const audio::AudioReactiveSystem::Settings& b) {
    return a.enabled == b.enabled &&
        a.sampleRate == b.sampleRate &&
        a.channels == b.channels &&
        a.fftSize == b.fftSize &&
        a.ringBufferMs == b.ringBufferMs &&
        a.keyCount == b.keyCount &&
        a.minFrequencyHz == b.minFrequencyHz &&
        a.maxFrequencyHz == b.maxFrequencyHz &&
        a.intensityGain == b.intensityGain &&
        a.noiseGateDb == b.noiseGateDb &&
        a.smoothingAttackSec == b.smoothingAttackSec &&
        a.smoothingReleaseSec == b.smoothingReleaseSec &&
        a.beatThreshold == b.beatThreshold &&
        a.beatHoldSeconds == b.beatHoldSeconds &&
        a.beatReleaseSeconds == b.beatReleaseSeconds &&
        a.globalEnergyEmaSeconds == b.globalEnergyEmaSeconds &&
        a.preferLoopback == b.preferLoopback &&
        a.fallbackToCapture == b.fallbackToCapture &&
        a.debugPrint == b.debugPrint;
}
static bool SeedAudioPoolScene(const console::RuntimeConsole& cc,
    const console::AudioPoolLayout& layout,
    sim::Simulator& simulator,
    sim::SimParams& simParams) {
    if (!layout.enabled || layout.total == 0)
        return false;
    simulator.seedBoxLattice(layout.nx, layout.ny, layout.nz, layout.poolMins, layout.spacing);
    simParams.numParticles = layout.total;
    return true;
}
static void UpdateAudioDebugOverlay(const console::RuntimeConsole& cc,
    const sim::SimParams& simParams,
    const sim::AudioFrameData* frameData,
    gfx::RendererD3D12& renderer) {
    bool debugEnabled = cc.audio.enabled && cc.audio.printDebug && (simParams.audio.enabled != 0);
    renderer.SetAudioDebugInfo(
        debugEnabled,
        simParams.audio.domainMinX,
        simParams.audio.invDomainWidth,
        simParams.audio.surfaceY,
        simParams.audio.surfaceFalloff,
        simParams.audio.keyCount);
    if (debugEnabled && frameData) {
        renderer.SetAudioDebugIntensities(frameData->keyIntensities, simParams.audio.keyCount);
    }
    else {
        renderer.SetAudioDebugIntensities(nullptr, 0);
    }
}
int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE, PWSTR, int) {
#ifdef ENABLE_NVTX
    nvtx3::scoped_range sr_main{ "Program.Main" };
#endif
    auto& cc = console::Instance();
    config::State cfgState{};
    {
        std::string cfgPath = MakeConfigAbsolutePath();
        std::string err;
        // 尝试 ?exe 同目录加载；失败则沿用默 ?cc 值继续运 ?
        if (!config::LoadFile(cfgPath, cc, &cfgState, &err) && !err.empty()) {
            std::fprintf(stderr, "Config load warning: %s\n", err.c_str());
        }
    }
    const console::AudioPoolLayout audioLayout = console::BuildAudioPoolLayout(cc);
    const bool useAudioDemo = audioLayout.enabled;
    audio::AudioReactiveSystem audioSystem;
    audio::AudioReactiveSystem::Settings activeAudioSettings = MakeAudioSettings(cc);
    bool poolDebugSweep = cc.audio.pool.debugKeySweep && useAudioDemo;
    bool audioSystemActive = false;
    if (!poolDebugSweep) {
        if (!audioSystem.initialize(activeAudioSettings) && activeAudioSettings.enabled) {
            std::fprintf(stderr, "[Audio] initialization failed, disabling audio-reactive forces.\n");
            cc.audio.enabled = false;
            activeAudioSettings.enabled = false;
        }
        else {
            audioSystemActive = activeAudioSettings.enabled;
        }
    }
    else {
        activeAudioSettings.enabled = false;
    }
    poolDebugSweep = poolDebugSweep && useAudioDemo;
    if (useAudioDemo && cc.audio.pool.overrideCamera) {
        cc.renderer.eye = cc.audio.pool.cameraEye;
        cc.renderer.at = cc.audio.pool.cameraAt;
        cc.renderer.up = make_float3(0.0f, 1.0f, 0.0f);
    }
    // 1) 应用窗口（使用配置中的分辨率/vsync ?
    HWND hwnd = CreateSimpleWindow(cc.app.width, cc.app.height, hInst);
    // 2) 渲染器初始化
    gfx::RendererD3D12 renderer;
    SetWindowLongPtr(hwnd, GWLP_USERDATA, (LONG_PTR)&renderer);
    gfx::RenderInitParams rp;
    console::BuildRenderInitParams(cc, rp);
    if (!renderer.Initialize(hwnd, rp)) return 1;

    console::ApplyRendererRuntime(cc, renderer);
    // Debug 初始化：默认开启，启动即暂 ?
    g_DebugEnabled = cc.debug.enabled;
    g_DebugPaused = cc.debug.pauseOnStart && g_DebugEnabled;
    g_DebugStepRequested = false;
    g_Camera = MakeCameraFromCc(cc);
    g_cam = &g_Camera;
    core::Profiler profiler;
    if (!useAudioDemo && cc.sim.demoMode == console::RuntimeConsole::Simulation::DemoMode::CubeMix) {
        console::PrepareCubeMix(cc); // 分解粒子与域拟合 + 颜色
    }
    // 3) 仿真初始 ?
    sim::Simulator simulator;
    sim::SimParams simParams{};
    console::BuildSimParams(cc, simParams);
    SanitizeRuntime(cc, simParams);
    g_Camera.moveSpeed = ComputeBaseSpeedFromDomain(simParams.grid.mins, simParams.grid.maxs);
    if (!simulator.initialize(simParams)) {
        MessageBoxW(hwnd, L"Simulator initialize failed.", L"PBF-X", MB_ICONERROR);
        return 1;
    }
    sim::UploadAudioForceParams(simParams.audio, simulator.cudaStream());
    const sim::AudioFrameData* initAudioFrame = nullptr;
    if (poolDebugSweep) {
        const auto& dbgFrame = NextPoolDemoDebugFrame(simParams.audio.keyCount, simParams.dt);
        sim::UploadAudioFrameData(dbgFrame, simulator.cudaStream());
        initAudioFrame = &dbgFrame;
    }
    else {
        const auto& audioFrame = audioSystem.frameData();
        sim::UploadAudioFrameData(audioFrame, simulator.cudaStream());
        initAudioFrame = audioSystemActive ? &audioFrame : nullptr;
    }
    UpdateAudioDebugOverlay(cc, simParams, initAudioFrame, renderer);
    auto& c = console::Instance();
    // 根据配置设置 NVTX
    prof::SetNvtxEnabled(c.perf.enable_nvtx);
    // 可选：单次标记启动阶段
    prof::Mark("App.Startup", prof::Color(0xFF, 0x80, 0x20));
    const uint32_t capacity = (simParams.maxParticles > 0) ? simParams.maxParticles : simParams.numParticles;
    // 移除旧的 cc.perf.use_external_pos_pingpong 标志：统一采用双缓 ?ping-pong
    HANDLE sharedA = nullptr, sharedB = nullptr;
    renderer.CreateSharedParticleBufferIndexed(0, capacity, sizeof(float4), sharedA);
    renderer.CreateSharedParticleBufferIndexed(1, capacity, sizeof(float4), sharedB);
    size_t bytes = size_t(capacity) * sizeof(float4);
    if (!simulator.bindExternalPosPingPong(sharedA, bytes, sharedB, bytes)) {
        std::fprintf(stderr, "[App][Error] bindExternalPosPingPong failed.\n");
    }
    CloseHandle(sharedA);
    CloseHandle(sharedB);
    HANDLE sharedVel = nullptr;
    if (renderer.CreateSharedVelocityBuffer(capacity, sizeof(float4), sharedVel)) {
        if (!simulator.bindExternalVelocityBuffer(sharedVel, bytes, sizeof(float4))) {
            std::fprintf(stderr, "[App][Warn] bindExternalVelocityBuffer failed.\n");
        }
    }
    else {
        std::fprintf(stderr, "[App][Warn] CreateSharedVelocityBuffer failed.\n");
    }
    if (sharedVel) CloseHandle(sharedVel);
    renderer.RegisterPingPongCudaPtrs(simulator.pingpongPosA(), simulator.pingpongPosB());
    renderer.SetParticleCount(simulator.activeParticleCount());
    // Timeline fence binding (zero CPU polling)
    if (!simulator.bindTimelineFence(renderer.SharedTimelineFenceHandle())) {
        std::fprintf(stderr, "[App][Warn] bindTimelineFence failed, fallback to sequential usage.\n");
    }
    // 5) 初始布点
    bool audioSeeded = false;
    if (useAudioDemo) {
        audioSeeded = SeedAudioPoolScene(cc, audioLayout, simulator, simParams);
        if (!audioSeeded) {
            std::fprintf(stderr, "[AudioDemo] SeedAudioPoolScene failed, fallback to CubeMix scene.\n");
        }
        else {
            renderer.SetParticleGrouping(0, 0);
        }
    }
    if (!useAudioDemo || !audioSeeded) {
        std::vector<float3> centers;
        console::GenerateCubeMixCenters(cc, centers);
        const uint32_t groups = cc.sim.cube_group_count;
        const uint32_t edge = cc.sim.cube_edge_particles;
        const float spacing = simParams.kernel.h *
            ((cc.sim.cube_lattice_spacing_factor_h > 0.f) ? cc.sim.cube_lattice_spacing_factor_h : 1.0f);
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

    uint64_t frameIndex = 0;
    MSG msg{};
    bool running = true;
    // 热加 ?诊断节流
    const uint32_t hotReloadEveryNFrames =
        (cc.app.hot_reload && cc.app.hot_reload_every_n > 0)
        ? (uint32_t)cc.app.hot_reload_every_n
        : UINT32_MAX;
    // 新：用于相机 dt（上一帧结 ? ?本帧开始）
    using Clock = std::chrono::steady_clock;
    static auto s_prevFrameEnd = Clock::now();
    // ================= 跑分状 ?=================
    const bool benchEnabled = cc.bench.enabled;
    const int benchTotalFrames = cc.bench.total_frames;
    const double benchTotalSeconds = cc.bench.total_seconds;
    const bool benchUseTimeRange = cc.bench.use_time_range;
    const int sampleBeginFrame = cc.bench.sample_begin_frame;
    const int sampleEndFrameCfg = cc.bench.sample_end_frame; //可能 <0
    const double sampleBeginSec = cc.bench.sample_begin_seconds;
    const double sampleEndSecCfg = cc.bench.sample_end_seconds;
    auto wallStart = Clock::now(); // 跑分整体起始时间
    // 累计数据（只统计采样窗口内）
    uint64_t benchSampledFrames = 0;
    double accumFrameMs = 0.0;
    double accumSimGpuMs = 0.0;
    double accumRenderMs = 0.0;
    double frameMsMin = 1e300, frameMsMax = 0.0;
    double simMsMin = 1e300, simMsMax = 0.0;
    double renderMsMin = 1e300, renderMsMax = 0.0;
    auto InFrameSampleWindow = [&](uint64_t frameIdx) -> bool {
        if (benchUseTimeRange) return false; // 帧范围模式下才走这里
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
    while (running) {
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) running = false;
            TranslateMessage(&msg); DispatchMessage(&msg);
        }
        // 帧开始时间（用于 frame_ms ?
        auto frameStart = Clock::now();
        double elapsedSec = std::chrono::duration<double>(frameStart - wallStart).count();
        // ==== 跑分结束条件判断（循环顶部即可早停）====
        if (benchEnabled) {
            bool reachedFrames = (benchTotalFrames > 0) && ((int64_t)frameIndex >= benchTotalFrames);
            bool reachedTime = (benchTotalSeconds > 0.0) && (elapsedSec >= benchTotalSeconds);
            if (reachedFrames || reachedTime) {
                running = false; //直接跳出主循 ?
            }
        }
        if (!running) break; // 避免多做一 ?
        // 热加 ?
        if (hotReloadEveryNFrames != UINT32_MAX &&
            (frameIndex % hotReloadEveryNFrames) == 0) {
            std::string err;
            if (config::TryHotReload(cfgState, cc, &err)) {
                console::BuildSimParams(cc, simParams);
                SanitizeRuntime(cc, simParams);
                console::ApplyRendererRuntime(cc, renderer);
                const sim::AudioFrameData* hotReloadFrame = poolDebugSweep ? &g_PoolDemoDebugFrame : (audioSystemActive ? &audioSystem.frameData() : nullptr);
                UpdateAudioDebugOverlay(cc, simParams, hotReloadFrame, renderer);
                auto newAudioSettings = MakeAudioSettings(cc);
                if (!AudioSettingsEqual(newAudioSettings, activeAudioSettings)) {
                    if (!poolDebugSweep) {
                        if (!audioSystem.initialize(newAudioSettings) && newAudioSettings.enabled) {
                            std::fprintf(stderr, "[Audio] reinitialize failed, disabling audio-reactive forces.\n");
                            cc.audio.enabled = false;
                            newAudioSettings.enabled = false;
                            simParams.audio.enabled = 0;
                        }
                        activeAudioSettings = newAudioSettings;
                        audioSystemActive = newAudioSettings.enabled;
                    }
                    else {
                        activeAudioSettings = newAudioSettings;
                        audioSystemActive = false;
                    }
                }

            }
            sim::UploadAudioForceParams(simParams.audio, simulator.cudaStream());
            if (cc.debug.printHotReload) {
                std::printf("[HotReload] Applied profile=%s K=%d maxN=%d sortN=%d point_size=%.2f\n",
                    cfgState.activeProfile.c_str(), simParams.solverIters, simParams.maxNeighbors, simParams.sortEveryN, cc.viewer.point_size_px);
            }
        }
    profiler.beginFrame(frameIndex);
    // 元数 ?
    profiler.addCounter("num_particles", (int64_t)simParams.numParticles);
    profiler.addCounter("solver_iters", simParams.solverIters);
    profiler.addCounter("max_neighbors", simParams.maxNeighbors);
    profiler.addCounter("sort_every_n", simParams.sortEveryN);
    profiler.addText("profile", cfgState.activeProfile);
    float dtSec = std::chrono::duration<float>(frameStart - s_prevFrameEnd).count();
    dtSec = std::clamp(dtSec, 0.0f, 0.2f);
    UpdateFreeFlyCamera(g_Camera, dtSec);
    SyncCameraToRenderer(cc, renderer, g_Camera);
    if (cc.audio.enabled) {
        HandleAudioTuningInput(cc, audioSystem);
    }
    const sim::AudioFrameData* frameForOverlay = nullptr;
    if (poolDebugSweep) {
        const auto& dbgFrame = NextPoolDemoDebugFrame(simParams.audio.keyCount, dtSec);
        sim::UploadAudioFrameData(dbgFrame, simulator.cudaStream());
        frameForOverlay = &dbgFrame;
    }
    else {
        if (audioSystemActive) {
            audioSystem.tick(dtSec);
        }
        const auto& audioFrame = audioSystem.frameData();
        sim::UploadAudioFrameData(audioFrame, simulator.cudaStream());
        frameForOverlay = audioSystemActive ? &audioFrame : nullptr;
    }
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
    bool doStep = true;
    if (g_DebugEnabled && !benchEnabled) {
        doStep = !g_DebugPaused || g_DebugStepRequested;
    }
    if (doStep) {
        sim::UploadAudioForceParams(simParams.audio, simulator.cudaStream());
        if (!simulator.step(simParams)) {
            std::fprintf(stderr, "[App][Error] simulator.step failed.\n");
        }
        renderer.UpdateParticleSRVForPingPong(simulator.renderPositionPtr());
        if (g_DebugEnabled && g_DebugStepRequested) {
            g_DebugStepRequested = false;
            g_DebugPaused = true;
        }
    }
    UpdateAudioDebugOverlay(cc, simParams, frameForOverlay, renderer);
    const double simMsGpu = simulator.lastGpuFrameMs();
    profiler.addRow("sim_ms_gpu", simMsGpu);
    renderer.SetParticleCount(simulator.activeParticleCount());
    renderer.WaitSimulationFence(simulator.lastSimFenceValue());
    auto renderStart = Clock::now();
    if (!simulator.externalPingPongEnabled()) {
        renderer.UpdateParticleSRVForPingPong(simulator.renderPositionPtr());
    }
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
    UpdateWindowTitleHUD(hwnd, cc, simulator.activeParticleCount(), s_fpsEma);
    if (benchEnabled) {
        bool inWindow = benchUseTimeRange ? InTimeSampleWindow(elapsedSec) : InFrameSampleWindow(frameIndex);
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
    // ======= 跑分结果输出 =======
    if (benchEnabled) {
        double benchElapsedSec = std::chrono::duration<double>(Clock::now() - wallStart).count();
        if (benchSampledFrames > 0) {
            double avgFrame = accumFrameMs / benchSampledFrames;
            double avgSim = accumSimGpuMs / benchSampledFrames;
            double avgRender = accumRenderMs / benchSampledFrames;
            double fps = 1000.0 / avgFrame;
            std::printf("\n==== Benchmark Result ====\n");
            std::printf("SampledFrames: %llu\n", (unsigned long long)benchSampledFrames);
            std::printf("TotalElapsedSec: %.3f\n", benchElapsedSec);
            if (!benchUseTimeRange) {
                std::printf("FrameRange: [%d, %s]\n", sampleBeginFrame,
                    (cc.bench.sample_end_frame >= 0) ? std::to_string(cc.bench.sample_end_frame).c_str() : "end");
            }
            else {
                std::printf("TimeRangeSec: [%.3f, %s]\n", sampleBeginSec,
                    (cc.bench.sample_end_seconds > 0.0) ? std::to_string(cc.bench.sample_end_seconds).c_str() : "end");
            }
            std::printf("Frame_ms avg=%.3f min=%.3f max=%.3f\n", avgFrame, frameMsMin, frameMsMax);
            //std::printf("Sim_msGpu avg=%.3f min=%.3f max=%.3f\n", avgSim, simMsMin, simMsMax);
            //std::printf("Render_ms avg=%.3f min=%.3f max=%.3f\n", avgRender, renderMsMin, renderMsMax);
            std::printf("FPS_avg=%.2f\n", fps);
            std::printf("==========================\n");
        }
        else {
            std::printf("[Benchmark] No frames sampled (check window configuration).\n");
        }
    }
    simulator.shutdown();
    renderer.Shutdown();
    audioSystem.shutdown();
    return 0;
}
int main() {
    return wWinMain(GetModuleHandleW(nullptr), nullptr, GetCommandLineW(), SW_SHOWNORMAL);
}
