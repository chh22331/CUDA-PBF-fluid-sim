#include <cstdio>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <windows.h>

#include "engine/core/config.h"
#include "engine/core/profiler.h"
#include "engine/gfx/renderer.h"

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_SIZE:
        if (wp != SIZE_MINIMIZED) {
            uint32_t w = LOWORD(lp), h = HIWORD(lp);
            // Store the new size in the GWLP_USERDATA bound renderer if available
            gfx::RendererD3D12* r = reinterpret_cast<gfx::RendererD3D12*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
            if (r) r->Resize(w, h);
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

// 改为调用 CPU 回退实现，避免与 CUDA 版本 run_vector_add 同名冲突
extern "C" int run_vector_add_cpu(const int* a, const int* b, int* c, unsigned int size);

int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE, PWSTR, int) {
    auto cfg = config::LoadDefault();
    HWND hwnd = CreateSimpleWindow(cfg.width, cfg.height, hInst);

    gfx::RendererD3D12 renderer;
    SetWindowLongPtr(hwnd, GWLP_USERDATA, (LONG_PTR)&renderer);

    gfx::RenderInitParams rp; rp.width = cfg.width; rp.height = cfg.height;
    if (!renderer.Initialize(hwnd, rp)) return 1;

    core::Profiler profiler;

    // quick sanity for CUDA fallback
    const unsigned size = 5; int a[size] = { 1,2,3,4,5 }; int b[size] = { 10,20,30,40,50 }; int c[size] = { 0 };
    run_vector_add_cpu(a, b, c, size);

    uint64_t frameIndex = 0;
    MSG msg{};
    bool running = true;
    while (running) {
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) running = false;
            TranslateMessage(&msg); DispatchMessage(&msg);
        }
        profiler.beginFrame(frameIndex);
        core::CpuTimer ft; ft.begin();

        renderer.RenderFrame(profiler);

        double frameMs = ft.endMs();
        profiler.addRow("frame", frameMs);
        profiler.flushCsv(cfg.csv_path, frameIndex);
        ++frameIndex;
    }

    renderer.Shutdown();
    return 0;
}

// Console 子系统下提供入口包装，转调 wWinMain
int main() {
    return wWinMain(GetModuleHandleW(nullptr), nullptr, GetCommandLineW(), SW_SHOWNORMAL);
}
