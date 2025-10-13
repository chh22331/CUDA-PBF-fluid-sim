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
        uint32_t groups;            // CubeMix: 立方体团数（非 CubeMix =0）
        uint32_t particlesPerGroup; // 单团粒子数（非 CubeMix =0）
        uint32_t pad0;
        uint32_t pad1;
    };

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

        bool ImportSharedBufferAsSRV(HANDLE sharedHandle, uint32_t numElements, uint32_t strideBytes, int& outSrvIndex);
        bool CreateSharedParticleBuffer(uint32_t numElements, uint32_t strideBytes, HANDLE& outSharedHandle);

        void SetCamera(const CameraParams& p) { m_camera = p; }
        void SetVisual(const VisualParams& v) { m_visual = v; }
        void SetParticleCount(uint32_t n) { m_particleCount = n; }
        void WaitForGPU();

        // —— 新增：上传分组调色板（RGB 连续） —— //
        bool UpdateGroupPalette(const float* rgbTriples, uint32_t groupCount);

        // —— 新增：设置分组元数据（CubeMix） —— //
        void SetParticleGrouping(uint32_t groups, uint32_t particlesPerGroup) {
            m_groups = groups;
            m_particlesPerGroup = particlesPerGroup;
        }

    private:
        void createThicknessResources(); // 保留
        void createNormalResources();    // 保留

        D3D12Device m_device;
        core::FrameGraph m_fg;

        float m_clearColor[4] = { 0.1f, 0.2f, 0.35f, 1.0f };

        Microsoft::WRL::ComPtr<ID3D12Resource> m_sharedParticleBuffer;
        int m_particleSrvIndex = -1;
        uint32_t m_particleCount = 0;

        // Palette
        Microsoft::WRL::ComPtr<ID3D12Resource> m_paletteBuffer;
        int m_paletteSrvIndex = -1;
        uint32_t m_groups = 0;
        uint32_t m_particlesPerGroup = 0;

        CameraParams m_camera{};
        VisualParams m_visual{};
    };

} // namespace gfx
