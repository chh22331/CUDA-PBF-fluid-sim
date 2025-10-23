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

    struct PerFrameCB {
        float viewProj[16];
        float screenSize[2];
        float particleRadiusPx;
        float thicknessScale;
        uint32_t groups;
        uint32_t particlesPerGroup;
        uint32_t pad0;
        uint32_t pad1;
    };

    struct CameraParams {
        float3 eye = make_float3(0.5f, 0.5f, 2.0f);
        float3 at = make_float3(0.5f, 0.5f, 0.5f);
        float3 up = make_float3(0.0f, 1.0f, 0.0f);
        float  fovYDeg = 45.0f;
        float  nearZ = 0.01f;
        float  farZ = 100.0f;
    };
    struct VisualParams {
        float particleRadiusPx = 3.0f;
        float thicknessScale = 1.0f;
        float clearColor[4] = { 0.1f, 0.2f, 0.35f, 1.0f };
    };

    class RendererD3D12 {
    public:
        bool Initialize(HWND hwnd, const RenderInitParams& p);
        void Resize(uint32_t w, uint32_t h) { m_device.resize(w, h); BuildFrameGraph(); }
        void BuildFrameGraph();
        void RenderFrame(core::Profiler& profiler);
        void Shutdown();

        // 原单缓冲接口（保留兼容）
        bool CreateSharedParticleBuffer(uint32_t numElements, uint32_t strideBytes, HANDLE& outSharedHandle);
        // 新增：带索引创建（slot=0/1），用于双外部 ping-pong
        bool CreateSharedParticleBufferIndexed(int slot, uint32_t numElements, uint32_t strideBytes, HANDLE& outSharedHandle);
        // 新增：根据当前 CUDA 设备指针切换显示 SRV
        void UpdateParticleSRVForPingPong(const void* devicePtrCurr);
        // 原单缓冲导入
        bool ImportSharedBufferAsSRV(HANDLE sharedHandle, uint32_t numElements, uint32_t strideBytes, int& outSrvIndex);
        // 新增：登记双缓冲 CUDA 设备指针（A=当前，B=下一帧），并立即尝试切换到 A
        void RegisterPingPongCudaPtrs(const void* ptrA, const void* ptrB);

        void SetCamera(const CameraParams& p) { m_camera = p; }
        void SetVisual(const VisualParams& v) { m_visual = v; }
        void SetParticleCount(uint32_t n) { m_particleCount = n; }
        void WaitForGPU();

        bool UpdateGroupPalette(const float* rgbTriples, uint32_t groupCount);
        void SetParticleGrouping(uint32_t groups, uint32_t particlesPerGroup) {
            m_groups = groups;
            m_particlesPerGroup = particlesPerGroup;
        }
        void SetUseHalfRender(bool enable){ m_useHalfRender = enable; }

        // 时间线同步访问器
        HANDLE SharedTimelineFenceHandle() const { return m_timelineFenceSharedHandle; }
        uint64_t CurrentRenderFenceValue() const { return m_renderFenceValue; }
        void WaitSimulationFence(uint64_t simValue); // 在渲染前等待模拟完成
        void SignalRenderComplete(uint64_t lastSimValue); // 在渲染结束后递增并 signal

        // 双缓冲资源
        Microsoft::WRL::ComPtr<ID3D12Resource> m_sharedParticleBuffers[2];
        int  m_particleSrvIndexPing[2] = { -1, -1 };
        void* m_knownCudaPtrs[2] = { nullptr, nullptr }; // 对应设备指针（cudaExternalMemory 映射）
        int  m_activePingIndex = 0;

    private:
        void createThicknessResources();
        void createNormalResources();

        D3D12Device m_device;
        core::FrameGraph m_fg;

        float m_clearColor[4] = { 0.1f, 0.2f, 0.35f, 1.0f };

        // 单缓冲旧字段（仍用于单外部预测）
        Microsoft::WRL::ComPtr<ID3D12Resource> m_sharedParticleBuffer;
        int m_particleSrvIndex = -1;

        uint32_t m_particleCount = 0;

        Microsoft::WRL::ComPtr<ID3D12Resource> m_paletteBuffer;
        int m_paletteSrvIndex = -1;
        uint32_t m_groups = 0;
        uint32_t m_particlesPerGroup = 0;

        CameraParams m_camera{};
        VisualParams m_visual{};

        // 新增：时间线 fence（与 CUDA external semaphore 共享）
        Microsoft::WRL::ComPtr<ID3D12Fence> m_timelineFence; // 单调递增：偶=render, 奇=sim 完成
        HANDLE m_timelineFenceSharedHandle = nullptr;
        uint64_t m_renderFenceValue = 0; // 最近一次渲染完成的偶数值
        uint64_t m_lastSimFenceValue =0; // 最近一次模拟完成的奇数值

        // 半精渲染链路：根据 precision.renderTransfer选择使用 half版着色器
        bool m_useHalfRender = false;
        Microsoft::WRL::ComPtr<ID3D12PipelineState> m_psoPointsFloat; // 原 float4版
        Microsoft::WRL::ComPtr<ID3D12PipelineState> m_psoPointsHalf; // half 压缩版 (uint2 解码)
    };

} // namespace gfx