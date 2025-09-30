#pragma once

#include <windows.h>
#include <wrl/client.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <cstdint>
#include <array>
#include <vector>

#ifndef SAFE_RELEASE
#define SAFE_RELEASE(p) if(p){ (p)->Release(); (p)=nullptr; }
#endif

namespace gfx {

using Microsoft::WRL::ComPtr;

struct DeviceInitParams {
    uint32_t width = 1280;
    uint32_t height = 720;
    DXGI_FORMAT backbufferFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
    uint32_t bufferCount = 3;
};

struct FrameResources {
    ComPtr<ID3D12CommandAllocator> cmdAllocator;
};

class D3D12Device {
public:
    bool initialize(HWND hwnd, const DeviceInitParams& p);
    void shutdown();

    bool beginFrame();
    void endFrame();

    void resize(uint32_t width, uint32_t height);

    ID3D12Device* device() const { return m_device.Get(); }
    ID3D12GraphicsCommandList* cmdList() const { return m_cmdList.Get(); }
    ID3D12CommandQueue* queue() const { return m_queue.Get(); }
    IDXGISwapChain3* swapchain() const { return m_swapchain.Get(); }

    uint32_t width() const { return m_params.width; }
    uint32_t height() const { return m_params.height; }

    uint32_t currentFrameIndex() const { return m_frameIndex; }
    D3D12_CPU_DESCRIPTOR_HANDLE currentRTV() const;
    ID3D12Resource* currentBackbuffer() const { return m_backbuffers[m_frameIndex].Get(); }

    void clearCurrentRTV(const float color[4]);
    void present();

    // GPU timestamp support (minimal: pair-wise pass timing)
    void resetTimestampCursor() { m_timestampCursor = 0; }
    void writeTimestamp();
    void resolveTimestamps();
    bool readbackPassTimesMs(std::vector<double>& outMs);

    // Descriptor heap for SRV and interop helpers
    bool createSrvHeap(uint32_t capacity, bool shaderVisible);
    int  createBufferSRV(ID3D12Resource* res, uint32_t numElements, uint32_t strideBytes);
    // Generic allocation and texture descriptors
    int  allocateSrvUavDescriptor();
    void createTextureSRVAtIndex(ID3D12Resource* res, DXGI_FORMAT fmt, int index);
    void createTextureUAVAtIndex(ID3D12Resource* res, DXGI_FORMAT fmt, int index);

    D3D12_CPU_DESCRIPTOR_HANDLE srvCpuHandleAt(uint32_t index) const;
    D3D12_GPU_DESCRIPTOR_HANDLE srvGpuHandleAt(uint32_t index) const;
    ID3D12DescriptorHeap* srvHeap() const { return m_srvHeap.Get(); }

    // Open shared resource (interop)
    bool openSharedResource(HANDLE h, REFIID riid, void** outPtr);

private:
    bool createDeviceAndSwapchain(HWND hwnd, const DeviceInitParams& p);
    bool createRTVHeapAndBuffers(uint32_t bufferCount, DXGI_FORMAT format);
    bool createCmdObjects(uint32_t bufferCount);
    bool createFence(uint32_t bufferCount);
    bool createTimestampObjects(uint32_t maxTimestampsPerFrame);
    void destroySwapchainResources();
    void waitForGPU();

private:
    DeviceInitParams m_params{};

    ComPtr<IDXGIFactory6> m_factory;
    ComPtr<IDXGISwapChain3> m_swapchain;
    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12CommandQueue> m_queue;

    ComPtr<ID3D12DescriptorHeap> m_rtvHeap;
    UINT m_rtvDescriptorSize = 0;

    // SRV heap (for interop SRVs)
    ComPtr<ID3D12DescriptorHeap> m_srvHeap;
    UINT m_srvDescriptorSize = 0;
    uint32_t m_srvCapacity = 0;
    uint32_t m_srvCount = 0;

    std::vector<ComPtr<ID3D12Resource>> m_backbuffers;
    std::vector<FrameResources> m_frames;
    ComPtr<ID3D12GraphicsCommandList> m_cmdList;

    // GPU timestamp
    ComPtr<ID3D12QueryHeap> m_queryHeap;
    ComPtr<ID3D12Resource>  m_queryReadback;
    uint32_t m_maxTimestampsPerFrame = 64;
    uint32_t m_timestampCursor = 0;
    uint32_t m_lastTimestampCount = 0;
    uint32_t m_lastCompletedFrameIndex = 0;

    ComPtr<ID3D12Fence> m_fence;
    HANDLE m_fenceEvent = nullptr;
    std::vector<UINT64> m_fenceValues;
    UINT64 m_lastSignaledFenceValue = 0;

    uint32_t m_frameIndex = 0;
};

}
