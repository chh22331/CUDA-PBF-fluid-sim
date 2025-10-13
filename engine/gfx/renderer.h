#pragma once
#include <cstdint>
#include <windows.h>
#include <string>
#include <vector>
#include "../core/frame_graph.h"
#include "../core/profiler.h"
#include "d3d12_device.h"
#include "../../sim/cuda_vec_math.cuh"

namespace gfx {

    struct RenderInitParams { uint32_t width = 1280, height = 720; bool vsync = false; };
    struct FrameView { float viewProj[16] = {}; };
    struct RenderInputGPU { uint64_t particlePositions = 0; uint32_t numParticles = 0; float particleRadiusWorld = 0.01f; };

    struct PerFrameCB {
        float viewProj[16];
        float screenSize[2];
        float particleRadiusPx;
        float thicknessScale;
    };

    // 新增：相机与可视化参数（由控制台注入）
    struct CameraParams {
        float3 eye = make_float3(0.5f, 0.5f, 2.0f);
        float3 at  = make_float3(0.5f, 0.5f, 0.5f);
        float3 up  = make_float3(0.0f, 1.0f, 0.0f);
        float  fovYDeg = 45.0f;
        float  nearZ = 0.01f;
        float  farZ  = 100.0f;
    };
    struct VisualParams {
        float particleRadiusPx = 3.0f;
        float thicknessScale   = 1.0f;
        float clearColor[4]    = {0.1f, 0.2f, 0.35f, 1.0f};
    };

    class RendererD3D12 {
    public:
        bool Initialize(HWND hwnd, const RenderInitParams& p);
        void Resize(uint32_t w, uint32_t h) { m_device.resize(w, h); BuildFrameGraph(); }
        void BuildFrameGraph();
        void RenderFrame(core::Profiler& profiler);
        void Shutdown();

        // Interop hooks
        bool ImportSharedBufferAsSRV(HANDLE sharedHandle, uint32_t numElements, uint32_t strideBytes, int& outSrvIndex);

        // 新增：创建共享粒子缓冲并返回共享句柄（CUDA 侧导入）；同时在本端注册为 SRV 以供绘制
        bool CreateSharedParticleBuffer(uint32_t numElements, uint32_t strideBytes, HANDLE& outSharedHandle);

        // 新增：由控制台注入相机/可视化参数
        void SetCamera(const CameraParams& p) { m_camera = p; }
        void SetVisual(const VisualParams& v) { m_visual = v; }

        // 新增：每帧更新实际要绘制的粒子数量（不得超过创建 SRV 时的 numElements）
        void SetParticleCount(uint32_t n) { m_particleCount = n; }
        void WaitForGPU();
    private:
        void addClearPresentPasses();
        void createThicknessResources();
        void createNormalResources();

    private:
        D3D12Device m_device;
        core::FrameGraph m_fg;

        // 移到 VisualParams 中的 clearColor，保留以兼容 BuildFrameGraph 实现
        float m_clearColor[4] = { 0.1f, 0.2f, 0.35f, 1.0f };

        // Interop state
        Microsoft::WRL::ComPtr<ID3D12Resource> m_sharedParticleBuffer;
        int m_particleSrvIndex = -1;
        uint32_t m_particleCount = 0;

        // Offscreen RTs（预留）
        Microsoft::WRL::ComPtr<ID3D12Resource> m_thicknessRT;
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_thicknessRtvHeap; // single RTV
        int m_thicknessSrvIndex = -1;

        Microsoft::WRL::ComPtr<ID3D12Resource> m_normalRT;
        int m_normalUavIndex = -1;
        int m_normalSrvIndex = -1;

        // 新增：运行时注入参数
        CameraParams m_camera{};
        VisualParams m_visual{};
    };
}
