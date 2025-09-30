#pragma once
#include <cstdint>
#include <windows.h>
#include <string>
#include <vector>
#include "../core/frame_graph.h"
#include "../core/profiler.h"
#include "d3d12_device.h"

namespace gfx {

struct RenderInitParams { uint32_t width = 1280, height = 720; };
struct FrameView { float viewProj[16] = {}; };
struct RenderInputGPU { uint64_t particlePositions = 0; uint32_t numParticles = 0; float particleRadiusWorld = 0.01f; };

struct PerFrameCB {
    float viewProj[16];
    float screenSize[2];
    float particleRadiusPx;
    float thicknessScale;
};

class RendererD3D12 {
public:
    bool Initialize(HWND hwnd, const RenderInitParams& p);
    void Resize(uint32_t w, uint32_t h) { m_device.resize(w,h); BuildFrameGraph(); }
    void BuildFrameGraph();
    void RenderFrame(core::Profiler& profiler);
    void Shutdown();

    // Interop hooks
    bool ImportSharedBufferAsSRV(HANDLE sharedHandle, uint32_t numElements, uint32_t strideBytes, int& outSrvIndex);

private:
    void addClearPresentPasses();
    void createThicknessResources();
    void createNormalResources();

private:
    D3D12Device m_device;
    core::FrameGraph m_fg;
    float m_clearColor[4] = {0.1f, 0.2f, 0.35f, 1.0f};

    // Interop state
    Microsoft::WRL::ComPtr<ID3D12Resource> m_sharedParticleBuffer;
    int m_particleSrvIndex = -1;
    uint32_t m_particleCount = 0;

    // Offscreen thickness RT (R16F)
    Microsoft::WRL::ComPtr<ID3D12Resource> m_thicknessRT;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_thicknessRtvHeap; // single RTV
    int m_thicknessSrvIndex = -1;

    // Normal RT (RG16F stored as float2 UAV)
    Microsoft::WRL::ComPtr<ID3D12Resource> m_normalRT;
    int m_normalUavIndex = -1;
    int m_normalSrvIndex = -1;
};

}
