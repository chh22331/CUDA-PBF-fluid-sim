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

        bool ImportSharedBufferAsSRV(HANDLE sharedHandle, uint32_t numElements, uint32_t strideBytes, int& outSrvIndex);

        void SetCamera(const CameraParams& p) { m_camera = p; }
        void SetVisual(const VisualParams& v) { m_visual = v; }
        void SetParticleCount(uint32_t n) { m_particleCount = n; }
        void WaitForGPU();

        bool UpdateGroupPalette(const float* rgbTriples, uint32_t groupCount);
        void SetParticleGrouping(uint32_t groups, uint32_t particlesPerGroup) {
            m_groups = groups;
            m_particlesPerGroup = particlesPerGroup;
        }
        // 双缓冲新增（slot 0/1）
        Microsoft::WRL::ComPtr<ID3D12Resource> m_sharedParticleBuffers[2];
        int  m_particleSrvIndexPing[2] = { -1, -1 };
        void* m_knownCudaPtrs[2] = { nullptr, nullptr }; // 来自 cudaExternalMemoryGetMappedBuffer 返回的设备指针
        int  m_activePingIndex = 0;

        // ==== 新增：仿真同步共享 Fence 接口 ===＝
        bool CreateSimulationSharedFence(HANDLE& outSharedHandle); // 创建共享 fence 供 CUDA 导入
        void AttachSimulationFenceFromHandle(HANDLE hSharedFence); // 可选：外部已创建时附加（未使用可忽略）
        void WaitSimulationFence(uint64_t value); // 在图形队列等待仿真 fence 达到 value
        uint64_t CurrentSimulationFenceValue() const { return m_simFenceValue; }
        void AdvanceSimulationFenceValue(uint64_t v) { m_simFenceValue = v; }

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

        // 仿真同步 fence（共享给 CUDA）
        Microsoft::WRL::ComPtr<ID3D12Fence> m_simFence; // D3D12_FENCE_FLAG_SHARED
        HANDLE m_simFenceSharedHandle = nullptr;        // 仅创建后临时持有供导入
        uint64_t m_simFenceValue = 0;                   // 最新已发出的等待值（跟踪方便）
    };

} // namespace gfx