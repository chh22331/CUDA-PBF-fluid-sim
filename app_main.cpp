#define NOMINMAX
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

#include "engine/core/config.h"
#include "engine/core/profiler.h"
#include "engine/gfx/renderer.h"
#include "engine/core/console.h"
#include "engine/core/prof_nvtx.h"
#include "sim/simulator.h"
#include "sim/parameters.h"
#include "sim/stats.h"

// ---- Debug 全局状态与工具 ----
static bool g_DebugEnabled = false;
static bool g_DebugPaused = false;
static bool g_DebugStepRequested = false;

// 使用 GetAsyncKeyState 轮询；只在 app_main.cpp 中包含 windows.h 已满足依赖
static inline bool KeyDown(int vk) {
    return (GetAsyncKeyState(vk) & 0x8000) != 0;
}

// 更新窗口标题栏（粒子数 + FPS）
static void UpdateWindowTitleHUD(HWND hwnd, uint32_t particleCount, double fps) {
    wchar_t buf[256];
    if (fps < 1e-3) fps = 0.0;
    swprintf(buf, L"PBF-X | N=%u | FPS=%.1f", particleCount, fps);
    SetWindowTextW(hwnd, buf);
}

// 上升沿触发（去抖）
static bool KeyPressedOnce(int vk) {
    static uint8_t prev[512] = {};
    vk &= 0x1FF; // 简单限界
    bool down = KeyDown(vk);
    bool fired = down && (prev[vk] == 0);
    prev[vk] = down ? 1 : 0;
    return fired;
}

// 简易 float3 工具
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
    double yaw = 0.0;    // 绕世界 Y
    double pitch = 0.0;  // 绕相机 X
    float  moveSpeed = 1.0f;        // 基础移动速度（单位/秒）
    float  mouseSensitivity = 0.002f; // 弧度/像素
};

static FreeFlyCamera g_Camera;
static FreeFlyCamera* g_cam = nullptr;
static bool  g_RmbDown = false;
static POINT g_LastMouse{};
static float g_PendingWheel = 0.0f;

// yaw/pitch -> 前向/右/上
static inline float3 CamForward(const FreeFlyCamera& c) {
    float cp = std::cos((float)c.pitch), sp = std::sin((float)c.pitch);
    float cy = std::cos((float)c.yaw), sy = std::sin((float)c.yaw);
    // yaw=0 指向 +Z；pitch 正向上
    return f3_norm(f3_make(sy * cp, sp, cy * cp));
}
static inline void CamBasis(const FreeFlyCamera& c, float3& fwd, float3& right, float3& up) {
    const float3 worldUp = f3_make(0.f, 1.f, 0.f);
    fwd = CamForward(c);
    right = f3_norm(f3_cross(fwd, worldUp));
    up = f3_cross(right, fwd);
}

// 从 cc 初始化相机（保持当前视角）
static FreeFlyCamera MakeCameraFromCc(const console::RuntimeConsole& cc) {
    FreeFlyCamera cam{};
    cam.position = cc.renderer.eye;
    float3 fwd = f3_norm(f3_sub(cc.renderer.at, cc.renderer.eye));
    // 反解 yaw/pitch
    cam.pitch = std::asin(std::clamp((double)fwd.y, -1.0, 1.0));
    cam.yaw = std::atan2((double)fwd.x, (double)fwd.z);
    cam.moveSpeed = 1.0f; // 启动后根据域尺寸再调整
    return cam;
}

// 根据域尺度设定基础速度（对 10x10x10 域给 ~3.0）
static float ComputeBaseSpeedFromDomain(const float3& mins, const float3& maxs) {
    float3 ext = f3_sub(maxs, mins);
    float  diag = f3_len(ext);
    return (((0.3f) > (diag * 0.3f)) ? (0.3f) : (diag * 0.3f));
}

// 每帧更新（键盘/滚轮），dtSec 用实时间隔
static void UpdateFreeFlyCamera(FreeFlyCamera& cam, float dtSec) {
    float3 fwd, right, up;
    CamBasis(cam, fwd, right, up);

    // 累积的滚轮推拉
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

// 将相机同步到渲染器（每帧调用）
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
            const double kLim = 1.5533; // ~89度
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

// 高开销“塌陷诊断”：打印受 cc.debug.printWarnings 控制，执行受 cc.debug.enableAdvancedCollapseDiag 控制
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

    // 子采样
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

    // 团簇半径（RMS / Max）
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

    // 阈值比例（越大越接近“塌陷/重叠”）
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

    // 人类可读的阈值报警（受 printWarnings 控制）
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
    // 迭代次数：1..64
    if (sp.solverIters < 1) { sp.solverIters = 1; if (log) std::fprintf(stderr, "[Sanitize] solverIters clamped to 1\n"); }
    if (sp.solverIters > 64) { sp.solverIters = 64; if (log) std::fprintf(stderr, "[Sanitize] solverIters clamped to 64\n"); }
    // 邻域上限：8..1024（并与 perf.neighbor_cap 一致）
    if (sp.maxNeighbors < 8) { sp.maxNeighbors = 8; if (log) std::fprintf(stderr, "[Sanitize] maxNeighbors clamped to 8\n"); }
    if (cc.perf.neighbor_cap > 0 && sp.maxNeighbors > cc.perf.neighbor_cap) {
        sp.maxNeighbors = cc.perf.neighbor_cap;
        if (log) std::fprintf(stderr, "[Sanitize] maxNeighbors clamped to perf.neighbor_cap=%d\n", cc.perf.neighbor_cap);
    }
    if (sp.maxNeighbors > 1024) { sp.maxNeighbors = 1024; if (log) std::fprintf(stderr, "[Sanitize] maxNeighbors clamped to 1024\n"); }
    // 排序频率：>=1
    if (sp.sortEveryN < 1) { sp.sortEveryN = 1; if (log) std::fprintf(stderr, "[Sanitize] sortEveryN clamped to 1\n"); }
    // 视图点大小：0.5..32
    if (cc.viewer.point_size_px < 0.5f) { cc.viewer.point_size_px = 0.5f; if (log) std::fprintf(stderr, "[Sanitize] viewer.point_size_px clamped to 0.5\n"); }
    if (cc.viewer.point_size_px > 32.f) { cc.viewer.point_size_px = 32.f; if (log) std::fprintf(stderr, "[Sanitize] viewer.point_size_px clamped to 32\n"); }
}

static void TickAndPrintSimStats(sim::Simulator& simulator, const sim::SimParams& simParams) {
    using Clock = std::chrono::steady_clock;
    using namespace std::chrono_literals;
    static auto last = Clock::now();
    static const auto interval = 2.0s;     // 每 2 秒打印一次
    static const uint32_t sampleStride = 2; // 采样步长，降低开销

    auto now = Clock::now();
    if (now - last < interval) return;
    last = now;

    sim::SimStats stats{};
    if (simulator.computeStats(stats, sampleStride)) {
        // 说明：avgRho 为基于 sumW 的估算值（ρ ≈ ρ0 * sumW）
        std::printf("[SimStats] N=%u, K=%d, maxN=%d, sortN=%d | "
            "avgNeighbors=%.2f, avgSpeed=%.4f, avgRhoRel=%.4f, avgRho=%.2f\n",
            stats.N,
            simParams.solverIters,
            simParams.maxNeighbors,
            simParams.sortEveryN,
            stats.avgNeighbors, stats.avgSpeed, stats.avgRhoRel, stats.avgRho);
        std::fflush(stdout);
    }
}

// 将相对路径 "config.json" 转换为与 exe 同目录的绝对路径
static std::string MakeConfigAbsolutePath() {
    wchar_t buf[MAX_PATH] = {};
    DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    std::wstring exePath(buf, (n ? n : 0));
    size_t slash = exePath.find_last_of(L"\\/");
    std::wstring dir = (slash == std::wstring::npos) ? L"." : exePath.substr(0, slash);
    std::wstring wfull = dir + L"\\config.json";
    // 转 UTF-8
    int len = WideCharToMultiByte(CP_UTF8, 0, wfull.c_str(), (int)wfull.size(), nullptr, 0, nullptr, nullptr);
    std::string out(len, '\0');
    WideCharToMultiByte(CP_UTF8, 0, wfull.c_str(), (int)wfull.size(), out.data(), len, nullptr, nullptr);
    return out;
}

// ================= 基准统计辅助 =================
struct BenchAccumulator {
    std::vector<double> simMs;    // 模拟耗时 (ms) （GPU优先，失败回退CPU）
    std::vector<double> renderMs; // 渲染耗时 (ms)
    void add(double s, double r) {
        if (s > 0.0) simMs.push_back(s); // 忽略 -1 / 0 无效 GPU 计时
        renderMs.push_back(r);
    }
};

static double Percentile(const std::vector<double>& v, double q) {
    if (v.empty()) return 0.0; if (q <= 0.0) return v.front(); if (q >= 1.0) return v.back();
    double pos = q * (v.size() - 1); size_t i0 = (size_t)std::floor(pos); size_t i1 = std::min(v.size() - 1, i0 + 1); double t = pos - i0; return v[i0] * (1.0 - t) + v[i1] * t;
}

static void PrintBenchSummary(const console::RuntimeConsole& cc, const BenchAccumulator& acc) {
    if (acc.renderMs.empty()) {
        std::printf("[Benchmark] No samples collected.\n");
        return;
    }
    std::vector<double> s = acc.simMs;
    std::vector<double> r = acc.renderMs;
    std::sort(r.begin(), r.end());
    if (!s.empty()) std::sort(s.begin(), s.end());

    auto avg = [](const std::vector<double>& a) {
        double sum = 0; for (double x : a) sum += x;
        return sum / std::max<size_t>(1, a.size());
        };

    double avgRender = avg(acc.renderMs);
    double p99Render = (r.empty() ? 0.0 : (r.size() == 1 ? r[0] : r[(size_t)std::round(0.99 * (r.size() - 1))]));
    double bestRender = r.empty() ? 0.0 : r.front();
    double worstRender = r.empty() ? 0.0 : r.back();

    bool hasSim = !s.empty();
    double avgSim = hasSim ? avg(acc.simMs) : 0.0;
    double p99Sim = hasSim ? (s.size() == 1 ? s[0] : s[(size_t)std::round(0.99 * (s.size() - 1))]) : 0.0;
    double bestSim = hasSim ? s.front() : 0.0;
    double worstSim = hasSim ? s.back() : 0.0;

    double avgFrame = hasSim ? (avgSim + avgRender) : avgRender;
    double fpsAvg = (avgFrame > 1e-6) ? 1000.0 / avgFrame : 0.0;
    double frameP99 = hasSim ? (p99Sim + p99Render) : p99Render;
    double fps1Low = (frameP99 > 1e-6) ? 1000.0 / frameP99 : 0.0;

    std::printf("================ Benchmark Summary ================\n");
    std::printf("Samples: %zu (frames %llu..%llu)\n",
        acc.renderMs.size(),
        (unsigned long long)cc.benchmark.sample_start,
        (unsigned long long)cc.benchmark.sample_end);

    if (hasSim) {
        std::printf("AvgSim=%.3f ms | AvgRender=%.3f ms | AvgFrame=%.3f ms | AvgFPS=%.2f\n",
            avgSim, avgRender, avgFrame, fpsAvg);
        std::printf("BestSim=%.3f ms | WorstSim=%.3f ms | 1%%LowSim(p99)=%.3f ms\n",
            bestSim, worstSim, p99Sim);
    }
    else {
        std::printf("No valid GPU sim timing collected (fallback only used for render).\n");
        std::printf("AvgRender=%.3f ms | AvgFrame≈Render=%.3f ms | AvgFPS=%.2f\n",
            avgRender, avgRender, fpsAvg);
    }
    std::printf("BestRender=%.3f ms | WorstRender=%.3f ms | 1%%LowRender(p99)=%.3f ms\n",
        bestRender, worstRender, p99Render);
    std::printf("1%%LowFPS~=%.2f (using frame p99 time)\n", fps1Low);
    std::printf("===================================================\n");
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
        // 尝试从 exe 同目录加载；失败则沿用默认 cc 值继续运行
        if (!config::LoadFile(cfgPath, cc, &cfgState, &err) && !err.empty()) {
            std::fprintf(stderr, "Config load warning: %s\n", err.c_str());
        }
    }
    // 如果基准模式强制 GPU 计时每帧
    if (cc.benchmark.enabled && cc.benchmark.force_full_gpu_timing) {
        cc.perf.frame_timing_every_n = 1; // 保证 Simulator 记录每帧 GPU sim 时间
    }

    // 1) 应用窗口（使用配置中的分辨率/vsync）
    HWND hwnd = CreateSimpleWindow(cc.app.width, cc.app.height, hInst);

    // 2) 渲染器初始化
    gfx::RendererD3D12 renderer;
    SetWindowLongPtr(hwnd, GWLP_USERDATA, (LONG_PTR)&renderer);

    gfx::RenderInitParams rp;
    console::BuildRenderInitParams(cc, rp);
    if (!renderer.Initialize(hwnd, rp)) return 1;

    // 注入相机/可视化运行时参数（清屏色/粒子半径等）
    console::FitCameraToDomain(cc);
    console::ApplyRendererRuntime(cc, renderer);
    // Debug 初始化：默认开启，启动即暂停
    g_DebugEnabled = cc.debug.enabled;
    g_DebugPaused = cc.debug.pauseOnStart && g_DebugEnabled;
    g_DebugStepRequested = false;
    g_Camera = MakeCameraFromCc(cc);
    g_Camera.moveSpeed = ComputeBaseSpeedFromDomain(cc.sim.gridMins, cc.sim.gridMaxs);
    g_cam = &g_Camera;

    core::Profiler profiler;
    if (cc.sim.demoMode == console::RuntimeConsole::Simulation::DemoMode::CubeMix) {
        console::PrepareCubeMix(cc); // 分解粒子与域拟合 + 颜色
    }

    // 3) 仿真初始化
    sim::Simulator simulator;
    sim::SimParams simParams{};
    console::BuildSimParams(cc, simParams);
    SanitizeRuntime(cc, simParams);
    if (!simulator.initialize(simParams)) {
        MessageBoxW(hwnd, L"Simulator initialize failed.", L"PBF-X", MB_ICONERROR);
        return 1;
    }
    auto& c = console::Instance();
    // 根据配置设置 NVTX
    prof::SetNvtxEnabled(c.perf.enable_nvtx);
    // 可选：单次标记启动阶段
    prof::Mark("App.Startup", prof::Color(0xFF, 0x80, 0x20));


    // 4) D3D12-CUDA 共享粒子缓冲
    {
        const uint32_t capacity = (simParams.maxParticles > 0) ? simParams.maxParticles : simParams.numParticles;
        if (cc.perf.use_external_pos_pingpong) {
            HANDLE sharedA = nullptr, sharedB = nullptr;
            renderer.CreateSharedParticleBufferIndexed(0, capacity, sizeof(float4), sharedA);
            renderer.CreateSharedParticleBufferIndexed(1, capacity, sizeof(float4), sharedB);

            size_t bytes = size_t(capacity) * sizeof(float4);
            if (!simulator.bindExternalPosPingPong(sharedA, bytes, sharedB, bytes)) {
                std::fprintf(stderr, "[App][Error] bindExternalPosPingPong failed.\n");
            }
            CloseHandle(sharedA);
            CloseHandle(sharedB);

            // 核心：登记两个设备指针（A=curr，B=next）
            renderer.RegisterPingPongCudaPtrs(simulator.pingpongPosA(), simulator.pingpongPosB());
            // 初始粒子数
            renderer.SetParticleCount(simulator.activeParticleCount());
        }
        else {
            HANDLE shared = nullptr;
            renderer.CreateSharedParticleBuffer(capacity, sizeof(float4), shared);
            simulator.importPosPredFromD3D12(shared, size_t(capacity) * sizeof(float4));
            CloseHandle(shared);
            renderer.SetParticleCount(simulator.activeParticleCount());
        }
    }

    // Timeline fence binding (zero CPU polling)
    if (!simulator.bindTimelineFence(renderer.SharedTimelineFenceHandle())) {
        std::fprintf(stderr, "[App][Warn] bindTimelineFence failed, fallback to sequential usage.\n");
    }

    // 5) 初始布点 - 根据模式选择
    if (cc.sim.demoMode == console::RuntimeConsole::Simulation::DemoMode::CubeMix) {
        // CubeMix 模式：生成立方体中心并为每个中心播种一个立方体
        std::vector<float3> centers;
        console::GenerateCubeMixCenters(cc, centers);
        const uint32_t groups = cc.sim.cube_group_count;
        const uint32_t edge = cc.sim.cube_edge_particles;
        const float spacing = simParams.kernel.h *
        ((cc.sim.cube_lattice_spacing_factor_h > 0.f) ? cc.sim.cube_lattice_spacing_factor_h : 1.0f);
            // 使用专用一次性播种接口（避免多次覆盖）
            simulator.seedCubeMix(groups,
                centers.data(),
                edge,
                spacing,
                (cc.sim.cube_apply_initial_jitter && cc.sim.initial_jitter_enable),
                cc.sim.initial_jitter_scale_h * simParams.kernel.h,
                cc.sim.initial_jitter_seed);
        // 上传调色板与分组元数据（供 VS/PS 使用）
        renderer.UpdateGroupPalette(&cc.sim.cube_group_colors[0][0], groups);
        renderer.SetParticleGrouping(groups, cc.sim.cube_particles_per_group);
    }
    else {
        // Faucet 模式：原有逻辑
        const float spacing = simParams.kernel.h *
            ((cc.sim.lattice_spacing_factor_h > 0.f) ? cc.sim.lattice_spacing_factor_h : 1.0f);
        const float3 origin = make_float3(simParams.grid.mins.x + 0.95f * spacing,
            simParams.grid.mins.y + 0.5f * spacing,
            simParams.grid.mins.z + 0.5f * spacing);
        simulator.seedBoxLatticeAuto(simParams.numParticles, origin, spacing);
        // 明确关闭分组（防止残留）
        renderer.UpdateGroupPalette(nullptr, 0);
        renderer.SetParticleGrouping(0, 0);
    }
    {
        const uint32_t active = simulator.activeParticleCount();
        simParams.numParticles = active;
        renderer.SetParticleCount(active);
    }

    uint64_t frameIndex = 0;
    MSG msg{};
    bool running = true;

    // 热加载/诊断节流
    const uint32_t hotReloadEveryNFrames =
        (cc.app.hot_reload && cc.app.hot_reload_every_n > 0)
        ? (uint32_t)cc.app.hot_reload_every_n
        : UINT32_MAX;
    const uint32_t diagEveryNFrames =
        (cc.debug.diag_every_n > 0)
        ? (uint32_t)cc.debug.diag_every_n
        : UINT32_MAX;

    // 新：用于相机 dt（上一帧结束 → 本帧开始）
    using Clock = std::chrono::steady_clock;
    static auto s_prevFrameEnd = Clock::now();

    // 基准统计容器
    BenchAccumulator benchAcc;

    while (running) {
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) running = false;
            TranslateMessage(&msg); DispatchMessage(&msg);
        }

        // 帧开始时间（用于 frame_ms）
        auto frameStart = Clock::now();

        // 热加载
        if (hotReloadEveryNFrames != UINT32_MAX &&
            (frameIndex % hotReloadEveryNFrames) == 0) {
            std::string err;
            if (config::TryHotReload(cfgState, cc, &err)) {
                console::BuildSimParams(cc, simParams);
                SanitizeRuntime(cc, simParams);
                console::FitCameraToDomain(cc);
                console::ApplyRendererRuntime(cc, renderer);
                if (cc.debug.printHotReload) {
                    std::printf("[HotReload] Applied profile=%s K=%d maxN=%d sortN=%d point_size=%.2f\n",
                        cfgState.activeProfile.c_str(), simParams.solverIters, simParams.maxNeighbors, simParams.sortEveryN, cc.viewer.point_size_px);
                }
            }
        }

        profiler.beginFrame(frameIndex);

        // 元数据
        profiler.addCounter("num_particles", (int64_t)simParams.numParticles);
        profiler.addCounter("solver_iters", simParams.solverIters);
        profiler.addCounter("max_neighbors", simParams.maxNeighbors);
        profiler.addCounter("sort_every_n", simParams.sortEveryN);
        profiler.addText("profile", cfgState.activeProfile);

        if (diagEveryNFrames != UINT32_MAX &&
            (frameIndex % diagEveryNFrames) == 0) {
            sim::SimStats sGrid{};
            if (simulator.computeStats(sGrid, 8)) {
                profiler.addCounter("avg_neighbors_grid_x1000", (int64_t)llround(sGrid.avgNeighbors * 1000.0));
                profiler.addCounter("avg_rho_rel_grid_x1000", (int64_t)llround(sGrid.avgRhoRel * 1000.0));
                profiler.addCounter("avg_rho_grid_x1000", (int64_t)llround(sGrid.avgRho * 1000.0));
            }

            sim::SimStats sBF{};
            if ((cc.debug.printDiagnostics || cc.debug.printWarnings || cc.debug.printHints) &&
                simulator.computeStatsBruteforce(sBF, 16, 512)) {
                profiler.addCounter("avg_neighbors_bf_x1000", (int64_t)llround(sBF.avgNeighbors * 1000.0));
                profiler.addCounter("avg_rho_rel_bf_x1000", (int64_t)llround(sBF.avgRhoRel * 1000.0));
                profiler.addCounter("avg_rho_bf_x1000", (int64_t)llround(sBF.avgRho * 1000.0));

                double nDiff = (sBF.avgNeighbors > 1e-9) ? (1.0 - sGrid.avgNeighbors / sBF.avgNeighbors) : 0.0;
                profiler.addCounter("neighbor_grid_vs_bf_diff_permille", (int64_t)llround(nDiff * 1000.0));
            }

            if (cc.debug.printWarnings) {
                // 阈值报警：接近上限（可能被截断）
                if (simParams.maxNeighbors > 0 && sGrid.avgNeighbors >= 0.9 * simParams.maxNeighbors) {
                    std::printf("[Warn] avgNeighbors_grid(%.1f) close to maxNeighbors(%d) -> risk of biased truncation.\n",
                        sGrid.avgNeighbors, simParams.maxNeighbors);
                }
                // 阈值报警：grid 明显低估邻居/密度（漏邻或截断）
                if (sBF.avgNeighbors > 0.0 && (sGrid.avgNeighbors < 0.8 * sBF.avgNeighbors)) {
                    std::printf("[Warn] Grid neighbor undercount: grid=%.1f, brute-force=%.1f. Check cellSize/h and early-stop.\n",
                        sGrid.avgNeighbors, sBF.avgNeighbors);
                }

                // 配置合理性检查
                if (simParams.grid.cellSize > 0.f && simParams.kernel.h > 0.f &&
                    simParams.grid.cellSize < simParams.kernel.h) {
                    std::printf("[Warn] grid.cellSize(%.3f) < h(%.3f): 27-cell search will miss neighbors.\n",
                        simParams.grid.cellSize, simParams.kernel.h);
                }
                if (simParams.sortEveryN > 1) {
                    std::printf("[Info] sortEveryN=%d may cause stale neighborhoods; reduce to 1 when debugging.\n", simParams.sortEveryN);
                }
            }

            // 质心与朝角点漂移
            static bool hasPrevCom = false;
            static float3 prevCom;
            float3 com{};
            if (simulator.computeCenterOfMass(com, /*sampleStride=*/8)) {
                profiler.addCounter("com_x_x1000", (int64_t)llround(com.x * 1000.0));
                profiler.addCounter("com_y_x1000", (int64_t)llround(com.y * 1000.0));
                profiler.addCounter("com_z_x1000", (int64_t)llround(com.z * 1000.0));

                if (hasPrevCom) {
                    float3 drift = make_float3(com.x - prevCom.x, com.y - prevCom.y, com.z - prevCom.z);
                    double driftLen = sqrt(drift.x * drift.x + drift.y * drift.y + drift.z * drift.z) + 1e-20;

                    // 选最近的角点
                    float3 corners[8] = {
                        simParams.grid.mins,
                        make_float3(simParams.grid.maxs.x, simParams.grid.mins.y, simParams.grid.mins.z),
                        make_float3(simParams.grid.mins.x, simParams.grid.maxs.y, simParams.grid.mins.z),
                        make_float3(simParams.grid.mins.x, simParams.grid.mins.y, simParams.grid.maxs.z),
                        make_float3(simParams.grid.maxs.x, simParams.grid.maxs.y, simParams.grid.mins.z),
                        make_float3(simParams.grid.maxs.x, simParams.grid.mins.y, simParams.grid.maxs.z),
                        make_float3(simParams.grid.mins.x, simParams.grid.maxs.y, simParams.grid.maxs.z),
                        simParams.grid.maxs
                    };
                    int nearest = 0; double bestD2 = 1e300;
                    for (int c = 0; c < 8; ++c) {
                        double dx = (double)corners[c].x - (double)com.x;
                        double dy = (double)corners[c].y - (double)com.y;
                        double dz = (double)corners[c].z - (double)com.z;
                        double d2 = dx * dx + dy * dy + dz * dz;
                        if (d2 < bestD2) { bestD2 = d2; nearest = c; }
                    }
                    float3 dir = make_float3(corners[nearest].x - com.x, corners[nearest].y - com.y, corners[nearest].z - com.z);
                    double dirLen = sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z) + 1e-20;
                    double cosTheta = ((double)drift.x * dir.x + (double)drift.y * dir.y + (double)drift.z * dir.z) / (driftLen * dirLen);

                    profiler.addCounter("com_drift_len_x1e6", (int64_t)llround(driftLen * 1e6));
                    profiler.addCounter("com_drift_toward_corner_cos_x1000", (int64_t)llround(cosTheta * 1000.0));

                    if (cc.debug.printWarnings) {
                        // 阈值报警：持续朝角点漂移（cos>0.8 且步长非零）
                        if (cosTheta > 0.8 && driftLen > 1e-6) {
                            std::printf("[Warn] COM drifts toward a corner: |drift|=%.3e, cos=%.2f (order bias or read/write race suspected).\n",
                                driftLen, cosTheta);
                        }
                    }
                }
                prevCom = com; hasPrevCom = true;
            }

            // 更详细的“塌陷诊断”日志与 CSV 指标（高开销，按需开启）
            if (cc.debug.enableAdvancedCollapseDiag) {
                LogAdvancedCollapseDiagnostics(cc, simulator, simParams, profiler);
            }
        }
        // 实时 dt（不依赖仿真 dt）
        using Clock = std::chrono::steady_clock;
        static auto tPrev = Clock::now();
        auto tNow = Clock::now();
        // 相机更新：用“上一帧结束→本帧开始”的 dt
        float dtSec = std::chrono::duration<float>(frameStart - s_prevFrameEnd).count();
        dtSec = std::clamp(dtSec, 0.0f, 0.2f);
        UpdateFreeFlyCamera(g_Camera, dtSec);
        SyncCameraToRenderer(cc, renderer, g_Camera);

        // Debug 控制
        if (g_DebugEnabled) {
            if (KeyPressedOnce(cc.debug.keyTogglePause)) {
                g_DebugPaused = !g_DebugPaused;
                if (cc.debug.printDebugKeys) {
                    std::printf("[Debug] %s\n", g_DebugPaused ? "Paused" : "Running");
                }
            }
            if (KeyPressedOnce(cc.debug.keyRun)) {
                g_DebugPaused = false;
                g_DebugStepRequested = false;
                if (cc.debug.printDebugKeys) {
                    std::printf("[Debug] Run (continuous)\n");
                }
            }
            if (KeyPressedOnce(cc.debug.keyStep)) {
                if (g_DebugPaused) {
                    g_DebugStepRequested = true;
                    if (cc.debug.printDebugKeys) {
                        std::printf("[Debug] Step requested\n");
                    }
                }
            }
        }

        // —— 模拟 —— //
        double simMsCpu = 0.0;
        bool doStep = (!g_DebugEnabled) || (!g_DebugPaused) || g_DebugStepRequested;
        if (doStep) {
            auto simStartWall = Clock::now();
            simulator.step(simParams);
            auto simEndWall = Clock::now();
            simMsCpu = std::chrono::duration<double, std::milli>(simEndWall - simStartWall).count();

            // 单步复位
            if (g_DebugEnabled && g_DebugStepRequested) {
                g_DebugStepRequested = false;
                g_DebugPaused = true;
            }

            // 条件 SRV 切换：只有真正发生指针交换时更新（最小化日志）
            if (simulator.externalPingPongEnabled() && simulator.swappedThisFrame()) {
                renderer.UpdateParticleSRVForPingPong(simulator.pingpongPosB()); ;
            }
        }
        // 保守冗余（可选）：若担心漏帧，可在此添加一次冪等调用
        // renderer.UpdateParticleSRVForPingPong(simulator.devicePositions());

        double simMsGpu = simulator.lastGpuFrameMs();
        double simMsEffective = (simMsGpu > 0.0) ? simMsGpu : simMsCpu;
        profiler.addRow("sim_ms_gpu", simMsGpu);
        profiler.addRow("sim_ms_effective", simMsEffective);

        // 更新粒子数（发射可能增加）
        renderer.SetParticleCount(simulator.activeParticleCount());

        // GPU timeline wait (non-blocking CPU): ensure simulation writes visible
        renderer.WaitSimulationFence(simulator.lastSimFenceValue());

        // —— 渲染 —— //
        auto renderStart = Clock::now();
        if (!simulator.externalPingPongEnabled()) {
            renderer.UpdateParticleSRVForPingPong(simulator.renderPositionPtr());
            std::fprintf(stderr, "[PP-Debug][Render] renderPtr=%p swapped=%d A=%p B=%p activeSrvIdx=%d\n",
                simulator.renderPositionPtr(),
                simulator.swappedThisFrame(),
                simulator.pingpongPosA(),
                simulator.pingpongPosB(),/* 可添加 renderer 当前 srvIndex */ 0 /* 替换为真实字段 */);
        } 
        renderer.RenderFrame(profiler);
        if (cc.perf.wait_render_gpu) { // 新性能开关决定是否保留 GPU 等待
            renderer.WaitForGPU();
        }
        // Signal render completion for optional future use
        renderer.SignalRenderComplete(simulator.lastSimFenceValue());
        auto renderEnd = Clock::now();
        const double renderMs = std::chrono::duration<double, std::milli>(renderEnd - renderStart).count();
        profiler.addRow("render_ms", renderMs);

        // —— 整帧耗时与 FPS —— //
        auto frameEnd = Clock::now();
        s_prevFrameEnd = frameEnd; // 下帧 dt 的参考
        const double frameMs = std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();
        const double fpsInst = (frameMs > 1e-6) ? (1000.0 / frameMs) : 0.0;

        static double s_fpsEma = 0.0;
        s_fpsEma = (s_fpsEma <= 0.0) ? fpsInst : (0.9 * s_fpsEma + 0.1 * fpsInst);

        profiler.addRow("frame_ms", frameMs);
        profiler.addRow("fps", fpsInst);

        UpdateWindowTitleHUD(hwnd, simParams.numParticles, s_fpsEma);

        if (cc.debug.printPeriodicStats) {
            TickAndPrintSimStats(simulator, simParams);
        }

        // ===== 基准统计采样 =====
        if (cc.benchmark.enabled) {
            uint64_t f = frameIndex;
            if (f >= cc.benchmark.sample_start && f <= cc.benchmark.sample_end) {
                benchAcc.add(simMsEffective, renderMs);
                if (cc.benchmark.print_each_frame) {
                    std::printf("[Bench][Frame=%llu] sim=%.3f ms (gpu=%.3f) render=%.3f ms\n",
                        (unsigned long long)f, simMsEffective, simMsGpu, renderMs);
                }
            }
            if (cc.benchmark.stop_after_steps > 0 && f + 1 >= cc.benchmark.stop_after_steps) {
                PrintBenchSummary(cc, benchAcc);
                running = false;
            }
        }

        profiler.flushCsv(cc.app.csv_path, frameIndex);
        ++frameIndex;
    }

    // 退出前如开启基准但未达 stop 条件也输出一次
    if (cc.benchmark.enabled) {
        PrintBenchSummary(cc, benchAcc);
    }

    simulator.shutdown();
    renderer.Shutdown();
    return 0;
}

int main() {
    return wWinMain(GetModuleHandleW(nullptr), nullptr, GetCommandLineW(), SW_SHOWNORMAL);
}