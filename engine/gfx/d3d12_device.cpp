#include "d3d12_device.h"
#include <cassert>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

namespace gfx {

bool D3D12Device::initialize(HWND hwnd, const DeviceInitParams& p) {
    m_params = p;
    if (!createDeviceAndSwapchain(hwnd, p)) return false;
    if (!createRTVHeapAndBuffers(p.bufferCount, p.backbufferFormat)) return false;
    if (!createCmdObjects(p.bufferCount)) return false;
    if (!createFence(p.bufferCount)) return false;
    if (!createTimestampObjects(m_maxTimestampsPerFrame)) return false;
    return true;
}

void D3D12Device::shutdown() {
    waitForGPU();
    if (m_fenceEvent) CloseHandle(m_fenceEvent);
    m_fenceEvent = nullptr;
}

bool D3D12Device::beginFrame() {
    auto& fr = m_frames[m_frameIndex];
    fr.cmdAllocator->Reset();
    m_cmdList->Reset(fr.cmdAllocator.Get(), nullptr);
    resetTimestampCursor();
    writeTimestamp(); // begin frame
    return true;
}

void D3D12Device::endFrame() {
    writeTimestamp(); // end frame
    m_cmdList->Close();
    ID3D12CommandList* lists[] = { m_cmdList.Get() };
    m_queue->ExecuteCommandLists(1, lists);
}

void D3D12Device::resize(uint32_t width, uint32_t height) {
    if (width == 0 || height == 0) return;
    waitForGPU();
    destroySwapchainResources();
    m_params.width = width; m_params.height = height;
    DXGI_SWAP_CHAIN_DESC sc{};
    m_swapchain->GetDesc(&sc);
    m_swapchain->ResizeBuffers(m_params.bufferCount, width, height, sc.BufferDesc.Format, sc.Flags);
    createRTVHeapAndBuffers(m_params.bufferCount, m_params.backbufferFormat);
    m_frameIndex = m_swapchain->GetCurrentBackBufferIndex();
}

D3D12_CPU_DESCRIPTOR_HANDLE D3D12Device::currentRTV() const {
    D3D12_CPU_DESCRIPTOR_HANDLE h = m_rtvHeap->GetCPUDescriptorHandleForHeapStart();
    h.ptr += SIZE_T(m_frameIndex) * SIZE_T(m_rtvDescriptorSize);
    return h;
}

void D3D12Device::clearCurrentRTV(const float color[4]) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = currentBackbuffer();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_cmdList->ResourceBarrier(1, &barrier);

    auto rtv = currentRTV();
    m_cmdList->ClearRenderTargetView(rtv, color, 0, nullptr);
}

void D3D12Device::present() {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = currentBackbuffer();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    m_cmdList->ResourceBarrier(1, &barrier);

    // Resolve timestamps before submitting command list
    resolveTimestamps();

    endFrame();

    // Drive vsync (syncInterval) directly from runtime config.
    m_swapchain->Present(m_params.vsync ? 1 : 0, 0);

    const UINT64 fenceToSignal = ++m_fenceValues[m_frameIndex];
    m_lastSignaledFenceValue = fenceToSignal;
    m_queue->Signal(m_fence.Get(), fenceToSignal);

    m_frameIndex = m_swapchain->GetCurrentBackBufferIndex();
    if (m_fence->GetCompletedValue() < m_fenceValues[m_frameIndex]) {
        m_fence->SetEventOnCompletion(m_fenceValues[m_frameIndex], m_fenceEvent);
        WaitForSingleObject(m_fenceEvent, INFINITE);
    }
}

bool D3D12Device::createDeviceAndSwapchain(HWND hwnd, const DeviceInitParams& p) {
    UINT factoryFlags = 0;
#if defined(_DEBUG)
    {
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
            debug->EnableDebugLayer();
            factoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
        }
    }
#endif

    if (FAILED(CreateDXGIFactory2(factoryFlags, IID_PPV_ARGS(&m_factory)))) return false;

    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0; m_factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&m_device)))) {
            break;
        }
    }
    if (!m_device) return false;

    D3D12_COMMAND_QUEUE_DESC qdesc{};
    qdesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    qdesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    if (FAILED(m_device->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&m_queue)))) return false;

    DXGI_SWAP_CHAIN_DESC1 scDesc{};
    scDesc.BufferCount = p.bufferCount;
    scDesc.Width = p.width;
    scDesc.Height = p.height;
    scDesc.Format = p.backbufferFormat;
    scDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    scDesc.SampleDesc.Count = 1;

    ComPtr<IDXGISwapChain1> swap1;
    if (FAILED(m_factory->CreateSwapChainForHwnd(m_queue.Get(), hwnd, &scDesc, nullptr, nullptr, &swap1))) return false;
    if (FAILED(swap1.As(&m_swapchain))) return false;

    m_frameIndex = m_swapchain->GetCurrentBackBufferIndex();
    return true;
}

bool D3D12Device::createRTVHeapAndBuffers(uint32_t bufferCount, DXGI_FORMAT format) {
    D3D12_DESCRIPTOR_HEAP_DESC rtvDesc{};
    rtvDesc.NumDescriptors = bufferCount;
    rtvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    if (FAILED(m_device->CreateDescriptorHeap(&rtvDesc, IID_PPV_ARGS(&m_rtvHeap)))) return false;
    m_rtvDescriptorSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    m_backbuffers.resize(bufferCount);
    for (uint32_t i = 0; i < bufferCount; ++i) {
        if (FAILED(m_swapchain->GetBuffer(i, IID_PPV_ARGS(&m_backbuffers[i])))) return false;
        D3D12_RENDER_TARGET_VIEW_DESC rtv{};
        rtv.Format = format;
        rtv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        auto handle = m_rtvHeap->GetCPUDescriptorHandleForHeapStart();
        handle.ptr += SIZE_T(i) * SIZE_T(m_rtvDescriptorSize);
        m_device->CreateRenderTargetView(m_backbuffers[i].Get(), &rtv, handle);
    }
    return true;
}

bool D3D12Device::createCmdObjects(uint32_t bufferCount) {
    m_frames.resize(bufferCount);

    for (uint32_t i = 0; i < bufferCount; ++i) {
        if (FAILED(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_frames[i].cmdAllocator)))) return false;
    }

    if (FAILED(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_frames[0].cmdAllocator.Get(), nullptr, IID_PPV_ARGS(&m_cmdList)))) return false;
    m_cmdList->Close();
    return true;
}

bool D3D12Device::createFence(uint32_t bufferCount) {
    if (FAILED(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)))) return false;
    m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!m_fenceEvent) return false;
    m_fenceValues.assign(bufferCount, 0);
    return true;
}

bool D3D12Device::createTimestampObjects(uint32_t maxTimestampsPerFrame) {
    D3D12_QUERY_HEAP_DESC qd{};
    qd.Count = maxTimestampsPerFrame;
    qd.NodeMask = 0;
    qd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    if (FAILED(m_device->CreateQueryHeap(&qd, IID_PPV_ARGS(&m_queryHeap)))) return false;

    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Alignment = 0;
    rd.Width = sizeof(UINT64) * maxTimestampsPerFrame;
    rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc = {1, 0};
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = D3D12_RESOURCE_FLAG_NONE;

    if (FAILED(m_device->CreateCommittedResource(
        &hp,
        D3D12_HEAP_FLAG_NONE,
        &rd,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&m_queryReadback)))) {
        return false;
    }
    return true;
}

void D3D12Device::destroySwapchainResources() {
    m_backbuffers.clear();
    m_rtvHeap.Reset();
}

void D3D12Device::writeTimestamp() {
    if (!m_queryHeap) return;
    if (m_timestampCursor >= m_maxTimestampsPerFrame) return;
    m_cmdList->EndQuery(m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, m_timestampCursor++);
}

void D3D12Device::resolveTimestamps() {
    if (!m_queryHeap || !m_queryReadback) return;
    if (m_timestampCursor == 0) return;
    m_lastTimestampCount = m_timestampCursor;
    m_cmdList->ResolveQueryData(m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, m_lastTimestampCount, m_queryReadback.Get(), 0);
}

bool D3D12Device::readbackPassTimesMs(std::vector<double>& outMs) {
    if (!m_queryReadback || m_lastTimestampCount < 2) return false;
    // Ensure GPU finished the last submitted frame that wrote timestamps
    if (m_fence->GetCompletedValue() < m_lastSignaledFenceValue) {
        m_fence->SetEventOnCompletion(m_lastSignaledFenceValue, m_fenceEvent);
        WaitForSingleObject(m_fenceEvent, INFINITE);
    }

    UINT64* mapped = nullptr;
    D3D12_RANGE r{0, sizeof(UINT64) * m_lastTimestampCount};
    if (FAILED(m_queryReadback->Map(0, &r, reinterpret_cast<void**>(&mapped)))) return false;

    UINT64 freq = 0;
    m_queue->GetTimestampFrequency(&freq);
    outMs.clear();
    for (uint32_t i = 0; i + 1 < m_lastTimestampCount; i += 2) {
        UINT64 dt = mapped[i+1] - mapped[i];
        double ms = (double)dt * 1000.0 / (double)freq;
        outMs.push_back(ms);
    }
    D3D12_RANGE w{0,0};
    m_queryReadback->Unmap(0, &w);
    return true;
}

bool D3D12Device::createSrvHeap(uint32_t capacity, bool shaderVisible) {
    D3D12_DESCRIPTOR_HEAP_DESC d{};
    d.NumDescriptors = capacity;
    d.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    d.Flags = shaderVisible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    if (FAILED(m_device->CreateDescriptorHeap(&d, IID_PPV_ARGS(&m_srvHeap)))) return false;
    m_srvDescriptorSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    m_srvCapacity = capacity; m_srvCount = 0;
    return true;
}

int D3D12Device::createBufferSRV(ID3D12Resource* res, uint32_t numElements, uint32_t strideBytes) {
    if (!m_srvHeap || m_srvCount >= m_srvCapacity) return -1;
    D3D12_SHADER_RESOURCE_VIEW_DESC sd{};
    sd.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    sd.Format = DXGI_FORMAT_UNKNOWN;
    sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    sd.Buffer.NumElements = numElements;
    sd.Buffer.StructureByteStride = strideBytes;
    sd.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    auto h = m_srvHeap->GetCPUDescriptorHandleForHeapStart();
    h.ptr += SIZE_T(m_srvCount) * SIZE_T(m_srvDescriptorSize);
    m_device->CreateShaderResourceView(res, &sd, h);
    return (int)m_srvCount++;
}

int D3D12Device::allocateSrvUavDescriptor() {
    if (!m_srvHeap || m_srvCount >= m_srvCapacity) return -1;
    return (int)m_srvCount++;
}

void D3D12Device::createTextureSRVAtIndex(ID3D12Resource* res, DXGI_FORMAT fmt, int index) {
    D3D12_SHADER_RESOURCE_VIEW_DESC sd{};
    sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    sd.Format = fmt;
    sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    sd.Texture2D.MostDetailedMip = 0;
    sd.Texture2D.MipLevels = 1;
    auto h = m_srvHeap->GetCPUDescriptorHandleForHeapStart();
    h.ptr += SIZE_T(index) * SIZE_T(m_srvDescriptorSize);
    m_device->CreateShaderResourceView(res, &sd, h);
}

void D3D12Device::createTextureUAVAtIndex(ID3D12Resource* res, DXGI_FORMAT fmt, int index) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC ud{};
    ud.Format = fmt;
    ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    auto h = m_srvHeap->GetCPUDescriptorHandleForHeapStart();
    h.ptr += SIZE_T(index) * SIZE_T(m_srvDescriptorSize);
    m_device->CreateUnorderedAccessView(res, nullptr, &ud, h);
}

D3D12_CPU_DESCRIPTOR_HANDLE D3D12Device::srvCpuHandleAt(uint32_t index) const {
    auto h = m_srvHeap->GetCPUDescriptorHandleForHeapStart();
    h.ptr += SIZE_T(index) * SIZE_T(m_srvDescriptorSize);
    return h;
}

D3D12_GPU_DESCRIPTOR_HANDLE D3D12Device::srvGpuHandleAt(uint32_t index) const {
    auto h = m_srvHeap->GetGPUDescriptorHandleForHeapStart();
    h.ptr += SIZE_T(index) * SIZE_T(m_srvDescriptorSize);
    return h;
}

bool D3D12Device::openSharedResource(HANDLE h, REFIID riid, void** outPtr) {
    if (!m_device) return false;
    return SUCCEEDED(m_device->OpenSharedHandle(h, riid, outPtr));
}

void D3D12Device::waitForGPU() {
    const UINT64 fenceToSignal = ++m_fenceValues[m_frameIndex];
    m_queue->Signal(m_fence.Get(), fenceToSignal);
    m_fence->SetEventOnCompletion(fenceToSignal, m_fenceEvent);
    WaitForSingleObject(m_fenceEvent, INFINITE);
}

}
