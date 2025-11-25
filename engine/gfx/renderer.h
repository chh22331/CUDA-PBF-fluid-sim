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
        enum class RenderMode {
            GroupPalette = 0,
            SpeedColor = 1
        };

        bool Initialize(HWND hwnd, const RenderInitParams& p);
        void Resize(uint32_t w, uint32_t h) { m_device.resize(w, h); BuildFrameGraph(); }
        void BuildFrameGraph();
        void RenderFrame(core::Profiler& profiler);
        void Shutdown();

        bool CreateSharedParticleBuffer(uint32_t numElements, uint32_t strideBytes, HANDLE& outSharedHandle);
        bool CreateSharedParticleBufferIndexed(int slot, uint32_t numElements, uint32_t strideBytes, HANDLE& outSharedHandle);
        void UpdateParticleSRVForPingPong(const void* devicePtrCurr);
        bool ImportSharedBufferAsSRV(HANDLE sharedHandle, uint32_t numElements, uint32_t strideBytes, int& outSrvIndex);
        void RegisterPingPongCudaPtrs(const void* ptrA, const void* ptrB);

        // 新增：导入速度共享缓冲作为 SRV（外层可把 CUDA 导出的 shared handle 传入）
        bool ImportSharedVelocityAsSRV(HANDLE sharedHandle, uint32_t numElements, uint32_t strideBytes, int& outSrvIndex);

        void SetCamera(const CameraParams& p) { m_camera = p; }
        void SetVisual(const VisualParams& v) { m_visual = v; }
        void SetParticleCount(uint32_t n) { m_particleCount = n; }
        void WaitForGPU();

        bool UpdateGroupPalette(const float* rgbTriples, uint32_t groupCount);
        void SetParticleGrouping(uint32_t groups, uint32_t particlesPerGroup) {
            m_groups = groups;
            m_particlesPerGroup = particlesPerGroup;
        }
        void SetUseHalfRender(bool enable) { m_useHalfRender = enable; }

        void SetRenderMode(RenderMode m) { m_renderMode = m; }

        // 时间线同步访问器
        HANDLE SharedTimelineFenceHandle() const { return m_timelineFenceSharedHandle; }
        uint64_t CurrentRenderFenceValue() const { return m_renderFenceValue; }
        void WaitSimulationFence(uint64_t simValue);
        void SignalRenderComplete(uint64_t lastSimValue);

        // 新增公开只读访问器（供 Unity 导出函数使用，避免非法访问 private)
        HANDLE SharedParticleBufferHandle(int slot) const {
            return (slot >= 0 && slot < 2) ? m_sharedParticleBufferHandles[slot] : nullptr;
        }
        int ActivePingIndex() const { return m_activePingIndex; }
        uint32_t ParticleCount() const { return m_particleCount; }

        Microsoft::WRL::ComPtr<ID3D12Resource> m_sharedParticleBuffers[2];
        int  m_particleSrvIndexPing[2] = { -1, -1 };
        void* m_knownCudaPtrs[2] = { nullptr, nullptr };
        int  m_activePingIndex = 0;

        HANDLE   m_timelineFenceSharedHandle = nullptr;
        uint64_t m_renderFenceValue = 0;
        uint64_t m_lastSimFenceValue = 0;
        D3D12Device m_device;
        uint32_t m_particleStrideBytes = 0;
        uint32_t m_groups = 0;
        uint32_t m_particlesPerGroup = 0;

    private:
        core::FrameGraph m_fg;
        float m_clearColor[4] = { 0.1f, 0.2f, 0.35f, 1.0f };

        Microsoft::WRL::ComPtr<ID3D12Resource> m_sharedParticleBuffer;
        int m_particleSrvIndex = -1;

        uint32_t m_particleCount = 0;

        Microsoft::WRL::ComPtr<ID3D12Resource> m_paletteBuffer;
        int m_paletteSrvIndex = -1;

        CameraParams m_camera{};
        VisualParams m_visual{};

        HANDLE m_sharedParticleBufferHandles[2]{};

        Microsoft::WRL::ComPtr<ID3D12Fence> m_timelineFence;


        bool m_useHalfRender = false;
        Microsoft::WRL::ComPtr<ID3D12PipelineState> m_psoPointsFloat;
        Microsoft::WRL::ComPtr<ID3D12PipelineState> m_psoPointsHalf;

        // 新增：速度渲染用资源与 PSO
        Microsoft::WRL::ComPtr<ID3D12Resource> m_sharedVelocityBuffer;
        int m_velocitySrvIndex = -1;
        Microsoft::WRL::ComPtr<ID3D12PipelineState> m_psoPointsSpeed;

        RenderMode m_renderMode = RenderMode::GroupPalette;
    };

} // namespace gfx