#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#pragma comment(lib, "d3d12.lib")

#include "IUnityInterface.h"
#include "IUnityGraphics.h"
#include "IUnityGraphicsD3D12.h"
#include "SimInterop.h"
#include "SimStatsInterop.h"

#include "../sim/simulator.h"
#include "../engine/core/console.h"

// ================= CUDA Driver 动态加载符号 =================
static HMODULE g_cudaDriverModule = nullptr;

typedef CUresult (CUDAAPI* PFN_cuInit)(unsigned int);
typedef CUresult (CUDAAPI* PFN_cuDeviceGetCount)(int*);
typedef CUresult (CUDAAPI* PFN_cuDeviceGet)(CUdevice*, int);
typedef CUresult (CUDAAPI* PFN_cuDeviceGetLuid)(char*, unsigned int*, CUdevice);
typedef CUresult (CUDAAPI* PFN_cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice);
typedef CUresult (CUDAAPI* PFN_cuDevicePrimaryCtxRelease)(CUdevice);
typedef CUresult (CUDAAPI* PFN_cuCtxSetCurrent)(CUcontext);
typedef CUresult (CUDAAPI* PFN_cuStreamCreate)(CUstream*, unsigned int);
typedef CUresult (CUDAAPI* PFN_cuStreamDestroy)(CUstream);

typedef CUresult (CUDAAPI* PFN_cuImportExternalMemory)(CUexternalMemory*, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC*);
typedef CUresult (CUDAAPI* PFN_cuExternalMemoryGetMappedBuffer)(CUdeviceptr*, CUexternalMemory, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC*);
typedef CUresult (CUDAAPI* PFN_cuDestroyExternalMemory)(CUexternalMemory);

typedef CUresult (CUDAAPI* PFN_cuImportExternalSemaphore)(CUexternalSemaphore*, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC*);
typedef CUresult (CUDAAPI* PFN_cuSignalExternalSemaphoresAsync)(const CUexternalSemaphore*, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS*, unsigned int, CUstream);
typedef CUresult (CUDAAPI* PFN_cuWaitExternalSemaphoresAsync)(const CUexternalSemaphore*, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS*, unsigned int, CUstream);
typedef CUresult (CUDAAPI* PFN_cuDestroyExternalSemaphore)(CUexternalSemaphore);

typedef CUresult (CUDAAPI* PFN_cuModuleLoadData)(CUmodule*, const void*);
typedef CUresult (CUDAAPI* PFN_cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
typedef CUresult (CUDAAPI* PFN_cuLaunchKernel)(CUfunction,
    unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int,
    unsigned int, CUstream, void**, void**);
typedef CUresult (CUDAAPI* PFN_cuGetErrorString)(CUresult, const char**);

static PFN_cuInit                       p_cuInit = nullptr;
static PFN_cuDeviceGetCount             p_cuDeviceGetCount = nullptr;
static PFN_cuDeviceGet                  p_cuDeviceGet = nullptr;
static PFN_cuDeviceGetLuid              p_cuDeviceGetLuid = nullptr;
static PFN_cuDevicePrimaryCtxRetain     p_cuDevicePrimaryCtxRetain = nullptr;
static PFN_cuDevicePrimaryCtxRelease    p_cuDevicePrimaryCtxRelease = nullptr;
static PFN_cuCtxSetCurrent              p_cuCtxSetCurrent = nullptr;
static PFN_cuStreamCreate               p_cuStreamCreate = nullptr;
static PFN_cuStreamDestroy              p_cuStreamDestroy = nullptr;

static PFN_cuImportExternalMemory       p_cuImportExternalMemory = nullptr;
static PFN_cuExternalMemoryGetMappedBuffer p_cuExternalMemoryGetMappedBuffer = nullptr;
static PFN_cuDestroyExternalMemory      p_cuDestroyExternalMemory = nullptr;

static PFN_cuImportExternalSemaphore    p_cuImportExternalSemaphore = nullptr;
static PFN_cuSignalExternalSemaphoresAsync p_cuSignalExternalSemaphoresAsync = nullptr;
static PFN_cuWaitExternalSemaphoresAsync p_cuWaitExternalSemaphoresAsync = nullptr;
static PFN_cuDestroyExternalSemaphore   p_cuDestroyExternalSemaphore = nullptr;

static PFN_cuModuleLoadData             p_cuModuleLoadData = nullptr;
static PFN_cuModuleGetFunction          p_cuModuleGetFunction = nullptr;
static PFN_cuLaunchKernel               p_cuLaunchKernel = nullptr;
static PFN_cuGetErrorString             p_cuGetErrorString = nullptr;

// CUDA 状态
static CUdevice  g_cuDevice = 0;
static CUcontext g_cuContext = nullptr;
static CUstream  g_cuStream = nullptr;
static bool      g_cudaAvailable = false;
static CUmodule  g_cuModule = 0;
static CUfunction g_cuFallbackKernel = 0;
static float     g_simTime = 0.0f;

// +++ GPU Timing: 事件与状态 +++
static cudaEvent_t g_evStart = nullptr;
static cudaEvent_t g_evEnd = nullptr;
static int   g_gpuTimingEnable = 0;
static float g_lastSimMs = -1.0f;
// --- GPU Timing 结束 ---

template<typename T>
static bool LoadSym(HMODULE m, T& out, const char* n1, const char* n2 = nullptr) {
    out = reinterpret_cast<T>(GetProcAddress(m, n1));
    if (!out && n2) out = reinterpret_cast<T>(GetProcAddress(m, n2));
    return out != nullptr;
}

static bool LoadCudaDriver() {
    if (g_cudaDriverModule) return true;
    g_cudaDriverModule = LoadLibraryA("nvcuda.dll");
    if (!g_cudaDriverModule) {
        OutputDebugStringA("[NativeSimPlugin][CUDA] LoadLibrary nvcuda.dll failed.\n");
        return false;
    }
    bool ok = true;
    ok &= LoadSym(g_cudaDriverModule, p_cuInit, "cuInit");
    ok &= LoadSym(g_cudaDriverModule, p_cuDeviceGetCount, "cuDeviceGetCount");
    ok &= LoadSym(g_cudaDriverModule, p_cuDeviceGet, "cuDeviceGet");
    ok &= LoadSym(g_cudaDriverModule, p_cuDeviceGetLuid, "cuDeviceGetLuid");
    ok &= LoadSym(g_cudaDriverModule, p_cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
    ok &= LoadSym(g_cudaDriverModule, p_cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2", "cuDevicePrimaryCtxRelease");
    ok &= LoadSym(g_cudaDriverModule, p_cuCtxSetCurrent, "cuCtxSetCurrent");
    ok &= LoadSym(g_cudaDriverModule, p_cuStreamCreate, "cuStreamCreate");
    ok &= LoadSym(g_cudaDriverModule, p_cuStreamDestroy, "cuStreamDestroy_v2", "cuStreamDestroy");
    ok &= LoadSym(g_cudaDriverModule, p_cuImportExternalMemory, "cuImportExternalMemory");
    ok &= LoadSym(g_cudaDriverModule, p_cuExternalMemoryGetMappedBuffer, "cuExternalMemoryGetMappedBuffer");
    ok &= LoadSym(g_cudaDriverModule, p_cuDestroyExternalMemory, "cuDestroyExternalMemory");
    ok &= LoadSym(g_cudaDriverModule, p_cuImportExternalSemaphore, "cuImportExternalSemaphore");
    ok &= LoadSym(g_cudaDriverModule, p_cuSignalExternalSemaphoresAsync, "cuSignalExternalSemaphoresAsync");
    ok &= LoadSym(g_cudaDriverModule, p_cuWaitExternalSemaphoresAsync, "cuWaitExternalSemaphoresAsync");
    ok &= LoadSym(g_cudaDriverModule, p_cuDestroyExternalSemaphore, "cuDestroyExternalSemaphore");
    ok &= LoadSym(g_cudaDriverModule, p_cuModuleLoadData, "cuModuleLoadData");
    ok &= LoadSym(g_cudaDriverModule, p_cuModuleGetFunction, "cuModuleGetFunction");
    ok &= LoadSym(g_cudaDriverModule, p_cuLaunchKernel, "cuLaunchKernel");
    LoadSym(g_cudaDriverModule, p_cuGetErrorString, "cuGetErrorString"); // 可选
    if (!ok) OutputDebugStringA("[NativeSimPlugin][CUDA] Symbol load incomplete.\n");
    return ok;
}

static void UnloadCudaDriver() {
    p_cuInit = nullptr;
    p_cuDeviceGetCount = nullptr;
    p_cuDeviceGet = nullptr;
    p_cuDeviceGetLuid = nullptr;
    p_cuDevicePrimaryCtxRetain = nullptr;
    p_cuDevicePrimaryCtxRelease = nullptr;
    p_cuCtxSetCurrent = nullptr;
    p_cuStreamCreate = nullptr;
    p_cuStreamDestroy = nullptr;
    p_cuImportExternalMemory = nullptr;
    p_cuExternalMemoryGetMappedBuffer = nullptr;
    p_cuDestroyExternalMemory = nullptr;
    p_cuImportExternalSemaphore = nullptr;
    p_cuSignalExternalSemaphoresAsync = nullptr;
    p_cuWaitExternalSemaphoresAsync = nullptr;
    p_cuDestroyExternalSemaphore = nullptr;
    p_cuModuleLoadData = nullptr;
    p_cuModuleGetFunction = nullptr;
    p_cuLaunchKernel = nullptr;
    p_cuGetErrorString = nullptr;
    if (g_cudaDriverModule) {
        FreeLibrary(g_cudaDriverModule);
        g_cudaDriverModule = nullptr;
    }
}

// ================= Unity / D3D12 基本状态 =================
static IUnityGraphicsD3D12* g_d3d12 = nullptr;
static ID3D12Device*        g_device = nullptr;
static ID3D12CommandQueue*  g_queue  = nullptr;

static ID3D12Fence* g_fence = nullptr;
static HANDLE       g_fenceSharedHandle = nullptr;
static UINT64       g_simFenceValue = 0;
static UINT64       g_lastWaitFence = 0;
static UINT64       g_fenceValue = 0;
static UINT64       g_lastSimSignal = 0;
static UINT64       g_lastRenderSignal = 0;
static int          g_enableCudaWait = 1;

// 双缓冲（Unity 侧 StructuredBuffer）
static ID3D12Resource* g_pingBuffers[2] = { nullptr, nullptr };
static HANDLE          g_pingSharedHandle[2] = { nullptr, nullptr };
static int  g_readIndex  = 0;
static int  g_writeIndex = 1;
static uint32_t g_particleStride   = 0;
static uint32_t g_particleCapacity = 0;
static uint32_t g_particleCount    = 0;

// 命令列表对象
static ID3D12CommandAllocator*    g_cmdAlloc = nullptr;
static ID3D12GraphicsCommandList* g_cmdList  = nullptr;

// External Memory / Semaphore
static CUexternalMemory   g_cuExtMem[2] = { 0, 0 };
static CUdeviceptr        g_cuDevPtr[2] = { 0, 0 };
static size_t             g_cuDevSize[2] = { 0, 0 };
static CUexternalSemaphore g_cuExtFence = 0;

// 模拟器
static sim::SimParams   g_cachedParams{};
static bool             g_paramsDirty = false;
static sim::Simulator*  g_sim = nullptr;
static bool             g_simulatorInitialized = false;

// ==================== 辅助 ====================
static void SafeRelease(IUnknown** pp) { if (pp && *pp) { (*pp)->Release(); *pp = nullptr; } }
static void SafeCloseHandle(HANDLE& h) { if (h) { CloseHandle(h); h = nullptr; } }

static void LogCudaError(const char* msg, CUresult rc) {
    const char* errStr = nullptr;
    if (p_cuGetErrorString && rc != CUDA_SUCCESS) p_cuGetErrorString(rc, &errStr);
    char buf[256];
    sprintf_s(buf, "[NativeSimPlugin][CUDA] %s rc=%d %s\n", msg ? msg : "CUDA Error", (int)rc, errStr ? errStr : "");
    OutputDebugStringA(buf);
}

static void Barrier(ID3D12GraphicsCommandList* cl, ID3D12Resource* res,
                    D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    if (!cl || !res || before == after) return;
    D3D12_RESOURCE_BARRIER b;
    std::memset(&b, 0, sizeof(b));
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = res;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    cl->ResourceBarrier(1, &b);
}

static void ExecCmdList() {
    if (!g_cmdList || !g_cmdAlloc || !g_queue) return;
    if (FAILED(g_cmdList->Close())) {
        OutputDebugStringA("[NativeSimPlugin] CommandList Close failed.\n");
        return;
    }
    ID3D12CommandList* lists[] = { g_cmdList };
    g_queue->ExecuteCommandLists(1, lists);
}

static void InitCommandObjects() {
    if (!g_device) return;
    if (!g_cmdAlloc) {
        if (FAILED(g_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&g_cmdAlloc)))) {
            OutputDebugStringA("[NativeSimPlugin] CreateCommandAllocator failed.\n");
            return;
        }
    }
    if (!g_cmdList) {
        if (FAILED(g_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g_cmdAlloc, nullptr, IID_PPV_ARGS(&g_cmdList)))) {
            OutputDebugStringA("[NativeSimPlugin] CreateCommandList failed.\n");
            return;
        }
        g_cmdList->Close();
    }
}

static bool CreateSharedFence() {
    if (!g_device) return false;
    SafeRelease(reinterpret_cast<IUnknown**>(&g_fence));
    SafeCloseHandle(g_fenceSharedHandle);
    if (FAILED(g_device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&g_fence)))) {
        OutputDebugStringA("[NativeSimPlugin] CreateFence failed.\n");
        return false;
    }
    if (FAILED(g_device->CreateSharedHandle(g_fence, nullptr, GENERIC_ALL, nullptr, &g_fenceSharedHandle))) {
        OutputDebugStringA("[NativeSimPlugin] CreateSharedHandle(Fence) failed.\n");
        SafeRelease(reinterpret_cast<IUnknown**>(&g_fence));
        return false;
    }
    g_simFenceValue = 0;
    g_fenceValue = 0;
    g_lastSimSignal = 0;
    g_lastRenderSignal = 0;
    g_lastWaitFence = 0;
    return true;
}

static bool GetAdapterLuid(LUID& outLuid) {
    if (!g_device) return false;
    IDXGIDevice* dxgiDev = nullptr;
    if (FAILED(g_device->QueryInterface(IID_PPV_ARGS(&dxgiDev)))) return false;
    IDXGIAdapter* adapter = nullptr;
    HRESULT hr = dxgiDev->GetAdapter(&adapter);
    dxgiDev->Release();
    if (FAILED(hr) || !adapter) return false;
    DXGI_ADAPTER_DESC desc;
    hr = adapter->GetDesc(&desc);
    adapter->Release();
    if (FAILED(hr)) return false;
    outLuid = desc.AdapterLuid;
    return true;
}

static CUresult MatchCudaDeviceByLuid(CUdevice* outDev, const LUID& luid) {
    if (!p_cuDeviceGetCount || !p_cuDeviceGet || !p_cuDeviceGetLuid)
        return CUDA_ERROR_NOT_INITIALIZED;
    int count = 0;
    CUresult rc = p_cuDeviceGetCount(&count);
    if (rc != CUDA_SUCCESS || count <= 0) return CUDA_ERROR_NO_DEVICE;
    for (int i = 0; i < count; ++i) {
        CUdevice cand;
        if (p_cuDeviceGet(&cand, i) != CUDA_SUCCESS) continue;
        char cuLuid[8] = {};
        unsigned int nodeMask = 0;
        if (p_cuDeviceGetLuid(cuLuid, &nodeMask, cand) == CUDA_SUCCESS) {
            if (std::memcmp(cuLuid, &luid, sizeof(LUID)) == 0) {
                *outDev = cand;
                return CUDA_SUCCESS;
            }
        }
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

// ================= External Memory =================
static void DestroyCudaExtForSlot(int slot) {
    if (slot < 0 || slot > 1) return;
    if (p_cuDestroyExternalMemory && g_cuExtMem[slot]) {
        p_cuDestroyExternalMemory(g_cuExtMem[slot]);
    }
    g_cuExtMem[slot] = 0;
    g_cuDevPtr[slot] = 0;
    g_cuDevSize[slot] = 0;
    SafeCloseHandle(g_pingSharedHandle[slot]);
}

static bool ImportCudaExtForSlot(int slot) {
    if (!g_cudaAvailable) return false;
    if (slot < 0 || slot > 1) return false;
    if (!g_device || !g_pingBuffers[slot]) return false;
    if (!p_cuImportExternalMemory || !p_cuExternalMemoryGetMappedBuffer) return false;

    if (g_cuExtMem[slot] && g_cuDevPtr[slot]) return true; // 已导入

    SafeCloseHandle(g_pingSharedHandle[slot]);
    HRESULT hr = g_device->CreateSharedHandle(g_pingBuffers[slot], nullptr, GENERIC_ALL, nullptr, &g_pingSharedHandle[slot]);
    if (FAILED(hr) || !g_pingSharedHandle[slot]) {
        OutputDebugStringA("[NativeSimPlugin][CUDA] CreateSharedHandle(Buffer) failed.\n");
        return false;
    }

    D3D12_RESOURCE_DESC desc = g_pingBuffers[slot]->GetDesc();
    UINT64 sizeBytes = desc.Width;

    CUDA_EXTERNAL_MEMORY_HANDLE_DESC mdesc{};
    mdesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
    mdesc.handle.win32.handle = g_pingSharedHandle[slot];
    mdesc.size = sizeBytes;
    mdesc.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;

    CUexternalMemory ext = 0;
    CUresult rc = p_cuImportExternalMemory(&ext, &mdesc);
    if (rc != CUDA_SUCCESS) {
        LogCudaError("cuImportExternalMemory", rc);
        SafeCloseHandle(g_pingSharedHandle[slot]);
        return false;
    }

    CUDA_EXTERNAL_MEMORY_BUFFER_DESC bdesc{};
    bdesc.offset = 0;
    bdesc.size = sizeBytes;
    bdesc.flags = 0;
    CUdeviceptr ptr = 0;
    rc = p_cuExternalMemoryGetMappedBuffer(&ptr, ext, &bdesc);
    if (rc != CUDA_SUCCESS || !ptr) {
        LogCudaError("cuExternalMemoryGetMappedBuffer", rc);
        if (p_cuDestroyExternalMemory) p_cuDestroyExternalMemory(ext);
        SafeCloseHandle(g_pingSharedHandle[slot]);
        return false;
    }

    g_cuExtMem[slot] = ext;
    g_cuDevPtr[slot] = ptr;
    g_cuDevSize[slot] = (size_t)sizeBytes;

    char buf[128];
    sprintf_s(buf, "[NativeSimPlugin][CUDA] Slot %d ExternalMemory mapped, %llu bytes.\n",
              slot, (unsigned long long)sizeBytes);
    OutputDebugStringA(buf);
    return true;
}

static bool ImportCudaFenceSemaphore() {
    if (!g_cudaAvailable || !g_fenceSharedHandle || !p_cuImportExternalSemaphore) return false;
    if (g_cuExtFence) return true;
    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC sdesc{};
    sdesc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE;
    sdesc.handle.win32.handle = g_fenceSharedHandle;
    sdesc.flags = 0;
    CUexternalSemaphore sem = 0;
    CUresult rc = p_cuImportExternalSemaphore(&sem, &sdesc);
    if (rc != CUDA_SUCCESS) {
        LogCudaError("cuImportExternalSemaphore(Fence)", rc);
        return false;
    }
    g_cuExtFence = sem;
    OutputDebugStringA("[NativeSimPlugin][CUDA] ExternalSemaphore(Fence) imported.\n");
    return true;
}

// ================= Fallback Kernel (stride=16) =================
// 仅在模拟器不可用时写简单位置
static const char* kFallbackPTX = R"ptx(
.version 7.0
.target sm_50
.address_size 64

.visible .entry FallbackKernel(
    .param .u64 bufPtr,
    .param .u32 count,
    .param .f32 time)
{
    .reg .pred  p;
    .reg .u32   tid, N, tmp;
    .reg .u64   base, ptr;
    .reg .f32   fIdx, fTime, fx, fy, fz;

    ld.param.u64 base, [bufPtr];
    ld.param.u32 N, [count];
    ld.param.f32 fTime, [time];

    mov.u32 tid, %tid.x;
    mad.lo.u32 tid, %ctaid.x, %ntid.x, tid;
    setp.ge.u32 p, tid, N;
    @p bra DONE;

    cvt.f32.u32 fIdx, tid;
    mul.f32 fx, fIdx, 0f3DCCCCCD;   // 0.1
    add.f32 fy, fTime, fIdx;
    mov.f32 fz, 0f00000000;

    mul.lo.u32 tmp, tid, 16;        // stride 16
    cvt.u64.u32 ptr, tmp;
    add.u64 ptr, base, ptr;

    // (x,y,z,w) w=组或半径，这里固定 0.05
    st.global.f32 [ptr+0], fx;
    st.global.f32 [ptr+4], fy;
    st.global.f32 [ptr+8], fz;
    st.global.f32 [ptr+12], 0f3D4CCCCD;

DONE:
    ret;
}
)ptx";

static void InitCudaKernelModule() {
    if (!g_cudaAvailable) return;
    if (!p_cuModuleLoadData || !p_cuModuleGetFunction) return;
    if (!g_cuModule) {
        CUresult rc = p_cuModuleLoadData(&g_cuModule, kFallbackPTX);
        if (rc != CUDA_SUCCESS) LogCudaError("cuModuleLoadData(Fallback)", rc);
    }
    if (g_cuModule && !g_cuFallbackKernel) {
        CUresult rc = p_cuModuleGetFunction(&g_cuFallbackKernel, g_cuModule, "FallbackKernel");
        if (rc != CUDA_SUCCESS) LogCudaError("cuModuleGetFunction(FallbackKernel)", rc);
        else OutputDebugStringA("[NativeSimPlugin][CUDA] FallbackKernel loaded.\n");
    }
}

// ================= CUDA 初始化/关闭 =================
static bool InitCuda() {
    if (!LoadCudaDriver()) return false;
    if (!p_cuInit || p_cuInit(0) != CUDA_SUCCESS) {
        OutputDebugStringA("[NativeSimPlugin][CUDA] cuInit failed.\n");
        return false;
    }
    LUID luid;
    if (!GetAdapterLuid(luid)) {
        OutputDebugStringA("[NativeSimPlugin][CUDA] GetAdapterLuid failed.\n");
        return false;
    }
    CUdevice dev = 0;
    if (MatchCudaDeviceByLuid(&dev, luid) != CUDA_SUCCESS) {
        OutputDebugStringA("[NativeSimPlugin][CUDA] LUID match failed.\n");
        return false;
    }
    if (!p_cuDevicePrimaryCtxRetain || !p_cuCtxSetCurrent || !p_cuStreamCreate) {
        OutputDebugStringA("[NativeSimPlugin][CUDA] Missing driver symbols.\n");
        return false;
    }
    CUcontext ctx = 0;
    if (p_cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS) {
        OutputDebugStringA("[NativeSimPlugin][CUDA] cuDevicePrimaryCtxRetain failed.\n");
        return false;
    }
    if (p_cuCtxSetCurrent(ctx) != CUDA_SUCCESS) {
        OutputDebugStringA("[NativeSimPlugin][CUDA] cuCtxSetCurrent failed.\n");
        if (p_cuDevicePrimaryCtxRelease) p_cuDevicePrimaryCtxRelease(dev);
        return false;
    }
    CUstream stm = 0;
    if (p_cuStreamCreate(&stm, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
        OutputDebugStringA("[NativeSimPlugin][CUDA] cuStreamCreate failed.\n");
        if (p_cuDevicePrimaryCtxRelease) p_cuDevicePrimaryCtxRelease(dev);
        return false;
    }
    g_cuDevice = dev;
    g_cuContext = ctx;
    g_cuStream = stm;
    g_cudaAvailable = true;
    OutputDebugStringA("[NativeSimPlugin][CUDA] Init OK.\n");

    ImportCudaFenceSemaphore();
    ImportCudaExtForSlot(0);
    ImportCudaExtForSlot(1);

    InitCudaKernelModule();
    return true;
}

static void ShutdownCuda() {
    if (p_cuDestroyExternalSemaphore && g_cuExtFence) {
        p_cuDestroyExternalSemaphore(g_cuExtFence);
        g_cuExtFence = 0;
    }
    DestroyCudaExtForSlot(0);
    DestroyCudaExtForSlot(1);
    if (g_cuModule) {
        g_cuModule = 0;
        g_cuFallbackKernel = 0;
    }
    // +++ GPU Timing：销毁事件 +++
    if (g_evStart) { cudaEventDestroy(g_evStart); g_evStart = nullptr; }
    if (g_evEnd) { cudaEventDestroy(g_evEnd);   g_evEnd = nullptr; }
    g_lastSimMs = -1.0f;
    g_gpuTimingEnable = 0;
    // --- GPU Timing 结束 ---
    if (p_cuStreamDestroy && g_cuStream) {
        p_cuStreamDestroy(g_cuStream);
        g_cuStream = 0;
    }
    if (p_cuCtxSetCurrent && p_cuDevicePrimaryCtxRelease && g_cuContext) {
        p_cuCtxSetCurrent(g_cuContext);
        p_cuDevicePrimaryCtxRelease(g_cuDevice);
        g_cuContext = 0;
        g_cuDevice = 0;
    }
    g_cudaAvailable = false;
    UnloadCudaDriver();
}

// ================= 模拟器导出接口 =================
extern "C" __declspec(dllexport) void Sim_SetParams(const SimParamsInterop* p) {
    if (!p) return;
    ConvertToNative(*p, g_cachedParams);
    g_paramsDirty = true;
    if (g_particleCapacity > 0) {
        if (g_cachedParams.maxParticles == 0 || g_cachedParams.maxParticles > g_particleCapacity)
            g_cachedParams.maxParticles = g_particleCapacity;
        if (g_cachedParams.numParticles > g_particleCapacity)
            g_cachedParams.numParticles = g_particleCapacity;
    }
}
extern "C" __declspec(dllexport) void Sim_GetParams(SimParamsInterop* out) {
    if (!out) return;
    ConvertToInterop(g_cachedParams, *out);
}
extern "C" __declspec(dllexport) void Sim_UpdateDt(float dt) {
    g_cachedParams.dt = dt; g_paramsDirty = true;
}
extern "C" __declspec(dllexport) void Sim_UpdateGravity(float gx, float gy, float gz) {
    g_cachedParams.gravity = make_float3(gx, gy, gz); g_paramsDirty = true;
}
extern "C" __declspec(dllexport) void Sim_UpdateCounts(uint32_t numParticles, uint32_t maxParticles) {
    g_cachedParams.numParticles = numParticles;
    if (maxParticles) g_cachedParams.maxParticles = maxParticles;
    g_paramsDirty = true;
}
extern "C" __declspec(dllexport) bool Sim_InitSimulator() {
    if (g_simulatorInitialized) return true;
    if (!g_cudaAvailable) {
        OutputDebugStringA("[NativeSimPlugin][Sim] CUDA unavailable.\n");
        return false;
    }
    if (!g_sim) g_sim = new sim::Simulator();
    if (g_cachedParams.kernel.h <= 0.f) g_cachedParams.kernel.h = 2.0f;
    g_cachedParams.kernel = sim::MakeKernelCoeffs(g_cachedParams.kernel.h);
    if (!g_sim->initialize(g_cachedParams)) {
        OutputDebugStringA("[NativeSimPlugin][Sim] Simulator initialize failed.\n");
        return false;
    }
    g_simulatorInitialized = true;
    OutputDebugStringA("[NativeSimPlugin][Sim] Simulator initialized.\n");
    return true;
}
extern "C" __declspec(dllexport) void Sim_ShutdownSimulator() {
    if (g_sim) {
        g_sim->shutdown();
        delete g_sim;
        g_sim = nullptr;
    }
    g_simulatorInitialized = false;
}
extern "C" __declspec(dllexport) bool Sim_SeedCubeMix(uint32_t groupCount,
                                                      uint32_t edgeParticles,
                                                      float spacing,
                                                      int jitterEnable,
                                                      float jitterAmp,
                                                      uint32_t jitterSeed) {
    if (!g_simulatorInitialized || !g_sim) return false;
    if (groupCount == 0 || edgeParticles == 0) return false;
    std::vector<float3> centers(groupCount);
    float baseX = 0.f;
    for (uint32_t g = 0; g < groupCount; ++g) {
        centers[g] = make_float3(baseX + g * spacing * edgeParticles * 1.5f,
                                 spacing * edgeParticles * 0.5f,
                                 0.0f);
    }
    g_sim->seedCubeMix(groupCount, centers.data(), edgeParticles, spacing,
                       jitterEnable != 0, jitterAmp, jitterSeed);
    g_cachedParams.numParticles = g_sim->activeParticleCount();
    OutputDebugStringA("[NativeSimPlugin][Sim] SeedCubeMix done.\n");
    return true;
}
extern "C" __declspec(dllexport) bool Sim_GetStats(SimStatsInterop* outStats) {
    if (!outStats) return false;
    if (!g_simulatorInitialized || !g_sim) {
        std::memset(outStats, 0, sizeof(*outStats));
        return false;
    }
    sim::SimStats st{};
    if (!g_sim->computeStats(st, 4)) return false;
    ConvertToInterop(st, *outStats);
    return true;
}
extern "C" __declspec(dllexport) void Sim_ApplyParamsNow() { g_paramsDirty = true; }
extern "C" __declspec(dllexport) void Sim_SetTime(float t) { g_simTime = t; }
extern "C" __declspec(dllexport) void Sim_EnableCudaWait(int enable) {
    g_enableCudaWait = (enable != 0) ? 1 : 0;
}

extern "C" __declspec(dllexport) void Sim_EnableGpuTiming(int enable) {
    int want = (enable != 0) ? 1 : 0;
    if (want == g_gpuTimingEnable) return;
    g_gpuTimingEnable = want;
    if (g_gpuTimingEnable) {
        if (!g_evStart) cudaEventCreateWithFlags(&g_evStart, cudaEventDefault);
        if (!g_evEnd)   cudaEventCreateWithFlags(&g_evEnd, cudaEventDefault);
        g_lastSimMs = -1.0f;
    }
    else {
        if (g_evStart) { cudaEventDestroy(g_evStart); g_evStart = nullptr; }
        if (g_evEnd) { cudaEventDestroy(g_evEnd);   g_evEnd = nullptr; }
        g_lastSimMs = -1.0f;
    }
}

extern "C" __declspec(dllexport) float Sim_GetLastSimGpuMs() {
    return g_lastSimMs;
}

// 绑定 Unity ping 缓冲（仅 stride/capacity 记录 + external memory 映射）
extern "C" __declspec(dllexport) void Sim_BindPingBuffer(int slot, void* nativeResPtr,
                                                         uint32_t strideBytes,
                                                         uint32_t capacityElements) {
    if (slot < 0 || slot > 1) return;
    ID3D12Resource* prev = g_pingBuffers[slot];
    g_pingBuffers[slot] = reinterpret_cast<ID3D12Resource*>(nativeResPtr);
    if (strideBytes) g_particleStride = strideBytes;
    if (capacityElements > g_particleCapacity) g_particleCapacity = capacityElements;
    if (prev != g_pingBuffers[slot]) DestroyCudaExtForSlot(slot);
    if (g_cudaAvailable && g_pingBuffers[slot]) ImportCudaExtForSlot(slot);
}

// 零拷贝绑定：让模拟器直接使用外部位置 ping-pong
extern "C" __declspec(dllexport) bool Sim_BindSimulatorExternalPosPingPong() {
    if (!g_simulatorInitialized || !g_sim) {
        OutputDebugStringA("[NativeSimPlugin][ZeroCopy] Simulator not initialized.\n");
        return false;
    }
    if (!g_device || !g_pingBuffers[0] || !g_pingBuffers[1]) {
        OutputDebugStringA("[NativeSimPlugin][ZeroCopy] Ping buffers not bound.\n");
        return false;
    }
    if (!g_cudaAvailable) {
        OutputDebugStringA("[NativeSimPlugin][ZeroCopy] CUDA unavailable.\n");
        return false;
    }
    auto EnsureSharedHandle = [](int slot)->bool {
        if (slot < 0 || slot > 1) return false;
        if (g_pingSharedHandle[slot]) return true;
        HRESULT hr = g_device->CreateSharedHandle(g_pingBuffers[slot], nullptr, GENERIC_ALL, nullptr, &g_pingSharedHandle[slot]);
        return SUCCEEDED(hr) && g_pingSharedHandle[slot];
    };
    if (!EnsureSharedHandle(0) || !EnsureSharedHandle(1)) return false;
    D3D12_RESOURCE_DESC d0 = g_pingBuffers[0]->GetDesc();
    D3D12_RESOURCE_DESC d1 = g_pingBuffers[1]->GetDesc();
    size_t bytes0 = (size_t)d0.Width;
    size_t bytes1 = (size_t)d1.Width;
    if (!g_sim->bindExternalPosPingPong((void*)g_pingSharedHandle[0], bytes0,
                                        (void*)g_pingSharedHandle[1], bytes1)) {
        OutputDebugStringA("[NativeSimPlugin][ZeroCopy] bindExternalPosPingPong failed.\n");
        return false;
    }
    OutputDebugStringA("[NativeSimPlugin][ZeroCopy] External pos ping-pong bound.\n");
    return true;
}

// 粒子数相关
extern "C" __declspec(dllexport) void     Sim_SetParticleCount(uint32_t count) { if (count > g_particleCapacity) count = g_particleCapacity; g_particleCount = count; }
extern "C" __declspec(dllexport) uint32_t Sim_GetParticleCount() { return g_particleCount; }
extern "C" __declspec(dllexport) uint32_t Sim_GetParticleStride() { return g_particleStride; }
extern "C" __declspec(dllexport) uint32_t Sim_GetParticleCapacity() { return g_particleCapacity; }
extern "C" __declspec(dllexport) int      Sim_GetReadIndex() { return g_readIndex; }
extern "C" __declspec(dllexport) int      Sim_GetWriteIndex() { return g_writeIndex; }
extern "C" __declspec(dllexport) uint64_t Sim_GetFenceValue() { return g_simFenceValue; }

// Tick：执行模拟或回退内核（零拷贝）
extern "C" __declspec(dllexport) void Sim_Tick() {
    // 等待上一帧渲染完成（双向 GPU 同步）
    if (g_enableCudaWait && g_cudaAvailable && g_cuExtFence && p_cuWaitExternalSemaphoresAsync && g_lastRenderSignal > 0) {
        CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS w{};
        w.params.fence.value = g_lastRenderSignal;
        p_cuWaitExternalSemaphoresAsync(&g_cuExtFence, &w, 1, g_cuStream);
    }

    // +++ GPU Timing：开始事件 +++
    if (g_gpuTimingEnable && g_evStart) {
        cudaEventRecord(g_evStart, g_cuStream);
    }

    bool usedSimulator = false;

    if (g_simulatorInitialized && g_sim) {
        if (g_paramsDirty) {
            if (g_cachedParams.kernel.h > 0.f)
                g_cachedParams.kernel = sim::MakeKernelCoeffs(g_cachedParams.kernel.h);
            g_paramsDirty = false;
        }
        if (g_particleCount > 0)
            g_cachedParams.numParticles = std::min<uint32_t>(g_particleCount, g_cachedParams.maxParticles);

        if (!g_sim->step(g_cachedParams)) {
            OutputDebugStringA("[NativeSimPlugin][Sim] step failed.\n");
        }
        else {
            usedSimulator = true;
            g_particleCount = g_sim->activeParticleCount();
        }
        // 零拷贝：模拟器已直接写外部 ping-pong，无需打包
    }

    // 回退占位（仅 stride>=16 时写 posRad）
    if (!usedSimulator) {
        if (g_cudaAvailable && g_cuFallbackKernel &&
            g_particleCount > 0 && g_cuDevPtr[g_writeIndex] != 0 &&
            g_particleStride >= 16) {
            CUdeviceptr buf = g_cuDevPtr[g_writeIndex];
            unsigned int count = g_particleCount;
            float timeParam = g_simTime;
            void* params[] = { &buf, &count, &timeParam };
            unsigned int block = 256;
            unsigned int grid = (count + block - 1) / block;
            CUresult rc = p_cuLaunchKernel(g_cuFallbackKernel,
                grid, 1, 1,
                block, 1, 1,
                0, g_cuStream,
                params, nullptr);
            if (rc != CUDA_SUCCESS) LogCudaError("FallbackKernel launch", rc);
        }
    }

    // +++ GPU Timing：结束事件 +++
    if (g_gpuTimingEnable && g_evEnd) {
        cudaEventRecord(g_evEnd, g_cuStream);
    }

    // 交换读写索引（与模拟器 ping-pong 保持一致）
    int newRead = g_writeIndex;
    int newWrite = g_readIndex;
    g_readIndex = newRead;
    g_writeIndex = newWrite;

    // 将写缓冲 Barrier 回 COMMON
    ID3D12Resource* writeRes = g_pingBuffers[g_writeIndex];
    if (writeRes && g_cmdAlloc && g_cmdList) {
        g_cmdAlloc->Reset();
        g_cmdList->Reset(g_cmdAlloc, nullptr);
        Barrier(g_cmdList, writeRes, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_COMMON);
        ExecCmdList();
    }

    // Signal fence
    ++g_fenceValue;
    g_simFenceValue = g_fenceValue;
    if (g_cudaAvailable && g_cuExtFence && p_cuSignalExternalSemaphoresAsync) {
        CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS s{};
        s.params.fence.value = g_fenceValue;
        CUresult rcs = p_cuSignalExternalSemaphoresAsync(&g_cuExtFence, &s, 1, g_cuStream);
        if (rcs != CUDA_SUCCESS && g_queue && g_fence)
            g_queue->Signal(g_fence, g_fenceValue);
    }
    else if (g_queue && g_fence) {
        g_queue->Signal(g_fence, g_fenceValue);
    }
    g_lastSimSignal = g_fenceValue;

    // +++ GPU Timing：尝试非阻塞读取耗时（若 end 事件尚未完成则保留上一次值） +++
    if (g_gpuTimingEnable && g_evStart && g_evEnd) {
        if (cudaEventQuery(g_evEnd) == cudaSuccess) {
            float ms = 0.0f;
            if (cudaEventElapsedTime(&ms, g_evStart, g_evEnd) == cudaSuccess) {
                g_lastSimMs = ms;
            }
        }
    }
}

// ============== Render Event ==============
using PFN_PLUGIN_EVENT = void(*)(int);
static void OnRenderEvent(int eventID) {
    if (eventID != 1) return;
    if (g_queue && g_fence && g_lastSimSignal > g_lastWaitFence) {
        g_queue->Wait(g_fence, g_lastSimSignal);
        g_lastWaitFence = g_lastSimSignal;
    }
    ID3D12Resource* readRes = g_pingBuffers[g_readIndex];
    if (readRes && g_cmdAlloc && g_cmdList) {
        g_cmdAlloc->Reset();
        g_cmdList->Reset(g_cmdAlloc, nullptr);
        Barrier(g_cmdList, readRes, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_GENERIC_READ);
        ExecCmdList();
    }
    if (g_queue && g_fence) {
        ++g_fenceValue;
        g_queue->Signal(g_fence, g_fenceValue);
        g_lastRenderSignal = g_fenceValue;
    }
}
extern "C" __declspec(dllexport) PFN_PLUGIN_EVENT GetRenderEventFunc() { return &OnRenderEvent; }

// ============== Unity 设备事件 ==============
static void UNITY_INTERFACE_API OnGfxEvent(UnityGfxDeviceEventType evt) {
    if (evt == kUnityGfxDeviceEventInitialize) {
        if (!g_d3d12) return;
        g_device = g_d3d12->GetDevice();
        g_queue  = g_d3d12->GetCommandQueue();
        CreateSharedFence();
        InitCommandObjects();
        OutputDebugStringA("[NativeSimPlugin] D3D12 Initialize.\n");
    } else if (evt == kUnityGfxDeviceEventShutdown) {
        SafeRelease(reinterpret_cast<IUnknown**>(&g_cmdList));
        SafeRelease(reinterpret_cast<IUnknown**>(&g_cmdAlloc));
        SafeRelease(reinterpret_cast<IUnknown**>(&g_fence));
        SafeCloseHandle(g_fenceSharedHandle);
        SafeCloseHandle(g_pingSharedHandle[0]);
        SafeCloseHandle(g_pingSharedHandle[1]);
        g_device = nullptr;
        g_queue  = nullptr;
        OutputDebugStringA("[NativeSimPlugin] D3D12 Shutdown.\n");
    }
}

// ============== 插件加载/卸载 ==============
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces* ifs) {
    IUnityGraphics* gfx = ifs->Get<IUnityGraphics>();
    g_d3d12 = ifs->Get<IUnityGraphicsD3D12>();
    if (gfx) gfx->RegisterDeviceEventCallback(OnGfxEvent);
}
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginUnload() {
    ShutdownCuda();
    // 其它资源在设备 Shutdown 回调中释放
}

// ============== 顶层初始化与关闭导出 ==============
extern "C" __declspec(dllexport) bool Sim_InitCuda() {
    if (!g_device) {
        OutputDebugStringA("[NativeSimPlugin] D3D12 device null.\n");
        return false;
    }
    return InitCuda();
}
extern "C" __declspec(dllexport) void Sim_ShutdownCuda() {
    ShutdownCuda();
}