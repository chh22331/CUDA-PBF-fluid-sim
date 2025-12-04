#include "renderer.h"
#include "../core/prof_nvtx.h"
#include "../../sim/cuda_vec_math.cuh"
#include <chrono>
#include <d3dcompiler.h>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#pragma comment(lib, "d3dcompiler.lib")

// Unity interop globals (exported handles and cached shared handles).
static gfx::RendererD3D12* g_unityExportRenderer = nullptr;
static HANDLE g_cachedSharedParticleHandles[2] = { nullptr, nullptr };

// Export: timeline fence shared handle for CUDA/Unity interop.
extern "C" __declspec(dllexport) uintptr_t Gfx_GetFenceHandle() {
    if (!g_unityExportRenderer) return 0;
    return reinterpret_cast<uintptr_t>(g_unityExportRenderer->m_timelineFenceSharedHandle);
}

// Export: particle ping buffer shared handle (lazily created).
extern "C" __declspec(dllexport) bool Gfx_GetPingBufferHandle(int slot, uintptr_t* outHandle) {
    if (!g_unityExportRenderer || !outHandle) return false;
    if (slot < 0 || slot > 1) return false;

    auto& res = g_unityExportRenderer->m_sharedParticleBuffers[slot];
    if (!res) return false;

    if (!g_cachedSharedParticleHandles[slot]) {
        HANDLE h = nullptr;
        if (FAILED(g_unityExportRenderer->m_device.device()->CreateSharedHandle(
                res.Get(), nullptr, GENERIC_ALL, nullptr, &h))) {
            return false;
        }
        g_cachedSharedParticleHandles[slot] = h;
    }
    *outHandle = reinterpret_cast<uintptr_t>(g_cachedSharedParticleHandles[slot]);
    return true;
}

// Export: active ping index (0/1), -1 if none.
extern "C" __declspec(dllexport) int Gfx_GetActivePingIndex() {
    if (!g_unityExportRenderer) return -1;
    return g_unityExportRenderer->ActivePingIndex();
}

// Export: particle count.
extern "C" __declspec(dllexport) uint32_t Gfx_GetParticleCount() {
    if (!g_unityExportRenderer) return 0;
    return g_unityExportRenderer->ParticleCount();
}

// Export: notify renderer about active CUDA device pointer to select SRV.
extern "C" __declspec(dllexport) void Gfx_SetActiveCudaDevicePtr(const void* p) {
    if (!g_unityExportRenderer) return;
    g_unityExportRenderer->UpdateParticleSRVForPingPong(p);
}

// Export: last simulation fence value recorded by renderer.
extern "C" __declspec(dllexport) uint64_t Gfx_GetLastSimFenceValue() {
    if (!g_unityExportRenderer) return 0;
    return g_unityExportRenderer->m_lastSimFenceValue;
}

// Export: particle stride (in bytes).
extern "C" __declspec(dllexport) uint32_t Gfx_GetParticleStride() {
    if (!g_unityExportRenderer) return 0;
    return g_unityExportRenderer->m_particleStrideBytes;
}

// Export: particle capacity across ping buffers.
extern "C" __declspec(dllexport) uint32_t Gfx_GetParticleCapacity() {
    if (!g_unityExportRenderer) return 0;
    uint32_t cap = 0;
    for (int i = 0; i < 2; ++i) {
        auto& r = g_unityExportRenderer->m_sharedParticleBuffers[i];
        if (r) {
            auto d = r->GetDesc();
            cap = std::max<uint32_t>(cap, (uint32_t)(d.Width / g_unityExportRenderer->m_particleStrideBytes));
        }
    }
    return cap;
}

// Export: update palette RGB triples (group-count sized).
extern "C" __declspec(dllexport) bool Gfx_UpdatePaletteRGB(const float* rgbTriples, uint32_t groupCount) {
    if (!g_unityExportRenderer) return false;
    return g_unityExportRenderer->UpdateGroupPalette(rgbTriples, groupCount);
}

// Export: update grouping info for palette mode.
extern "C" __declspec(dllexport) void Gfx_SetGroups(uint32_t groups, uint32_t particlesPerGroup) {
    if (!g_unityExportRenderer) return;
    g_unityExportRenderer->m_groups = groups;
    g_unityExportRenderer->m_particlesPerGroup = particlesPerGroup;
}

// Export: switch render mode (0 = GroupPalette, 1 = SpeedColor).
extern "C" __declspec(dllexport) void Gfx_SetRenderMode(int mode) {
    if (!g_unityExportRenderer) return;
    if (mode == 1) {
        g_unityExportRenderer->SetRenderMode(gfx::RendererD3D12::RenderMode::SpeedColor);
    } else {
        g_unityExportRenderer->SetRenderMode(gfx::RendererD3D12::RenderMode::GroupPalette);
    }
}

// Export: import external velocity buffer as SRV (returns descriptor index).
extern "C" __declspec(dllexport) bool Gfx_ImportVelocityShared(uintptr_t sharedHandle, uint32_t numElements, uint32_t strideBytes, int* outSrvIndex) {
    if (!g_unityExportRenderer || !outSrvIndex) return false;
    HANDLE h = reinterpret_cast<HANDLE>(sharedHandle);
    int idx = -1;
    bool ok = g_unityExportRenderer->ImportSharedVelocityAsSRV(h, numElements, strideBytes, idx);
    if (ok) *outSrvIndex = idx;
    return ok;
}

// Export: reset cached shared handle mirror (on resource rebuild).
extern "C" __declspec(dllexport) void Gfx_ResetSharedHandleCache() {
    for (int i = 0; i < 2; ++i) {
        if (g_cachedSharedParticleHandles[i]) {
            CloseHandle(g_cachedSharedParticleHandles[i]);
            g_cachedSharedParticleHandles[i] = nullptr;
        }
    }
}

namespace gfx {
namespace {

// Minimal 4x4 matrix for camera transforms.
struct Mat4 {
    float m[16];
};

static Mat4 mul(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int c = 0; c < 4; ++c) {
        for (int r0 = 0; r0 < 4; ++r0) {
            r.m[c * 4 + r0] =
                a.m[0 * 4 + r0] * b.m[c * 4 + 0] +
                a.m[1 * 4 + r0] * b.m[c * 4 + 1] +
                a.m[2 * 4 + r0] * b.m[c * 4 + 2] +
                a.m[3 * 4 + r0] * b.m[c * 4 + 3];
        }
    }
    return r;
}

static Mat4 lookAtRH(float3 eye, float3 at, float3 up) {
    float3 z = make_float3(eye.x - at.x, eye.y - at.y, eye.z - at.z);
    float lenz = sqrtf(z.x * z.x + z.y * z.y + z.z * z.z);
    z.x /= lenz; z.y /= lenz; z.z /= lenz;

    float3 x = make_float3(up.y * z.z - up.z * z.y, up.z * z.x - up.x * z.z, up.x * z.y - up.y * z.x);
    float lenx = sqrtf(x.x * x.x + x.y * x.y + x.z * x.z);
    x.x /= lenx; x.y /= lenx; x.z /= lenx;

    float3 y = make_float3(z.y * x.z - z.z * x.y, z.z * x.x - z.x * x.z, z.x * x.y - z.y * x.x);

    Mat4 m{};
    m.m[0] = x.x;  m.m[4] = x.y;  m.m[8]  = x.z;  m.m[12] = -(x.x * eye.x + x.y * eye.y + x.z * eye.z);
    m.m[1] = y.x;  m.m[5] = y.y;  m.m[9]  = y.z;  m.m[13] = -(y.x * eye.x + y.y * eye.y + y.z * eye.z);
    m.m[2] = z.x;  m.m[6] = z.y;  m.m[10] = z.z;  m.m[14] = -(z.x * eye.x + z.y * eye.y + z.z * eye.z);
    m.m[3] = 0.f;  m.m[7] = 0.f;  m.m[11] = 0.f;  m.m[15] = 1.f;
    return m;
}

static Mat4 perspectiveFovRH_ZO(float fovy, float aspect, float zn, float zf) {
    float f = 1.0f / tanf(fovy * 0.5f);
    Mat4 m{};
    m.m[0] = f / aspect;
    m.m[5] = f;
    m.m[10] = zf / (zn - zf);
    m.m[14] = (zf * zn) / (zn - zf);
    m.m[11] = -1.0f;
    return m;
}

static float Deg2Rad(float deg) {
    return deg * 3.14159265358979323846f / 180.0f;
}

// Simple path helpers (Unicode).
static std::wstring GetExeDirW() {
    wchar_t path[MAX_PATH] = {};
    DWORD n = GetModuleFileNameW(nullptr, path, MAX_PATH);
    std::wstring exe(path, n ? n : 0);
    size_t slash = exe.find_last_of(L"\\/");
    if (slash == std::wstring::npos) return L".";
    return exe.substr(0, slash);
}

static std::wstring GetCwdW() {
    DWORD n = GetCurrentDirectoryW(0, nullptr);
    std::wstring out;
    if (n > 0) {
        out.resize(n);
        GetCurrentDirectoryW(n, &out[0]);
        if (!out.empty() && out.back() == L'\0') out.pop_back();
    }
    return out;
}

static bool FileExistsW(const std::wstring& p) {
    DWORD a = GetFileAttributesW(p.c_str());
    return (a != INVALID_FILE_ATTRIBUTES) && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

static void ToBackslashes(std::wstring& s) {
    for (auto& ch : s) {
        if (ch == L'/') ch = L'\\';
    }
}

// Resolve shader file path by probing common locations in dev/runtime.
static std::wstring ResolveShaderPath(const wchar_t* relative) {
    std::wstring rel = relative ? std::wstring(relative) : std::wstring();
    ToBackslashes(rel);

    // If already absolute, return as-is.
    if (!rel.empty() && (rel[0] == L'\\' || (rel.size() > 1 && rel[1] == L':'))) {
        return rel;
    }

    // Candidate bases: exe dir, repo root (walk-up), cwd, source dir.
    std::vector<std::wstring> bases;
    const std::wstring exeDir = GetExeDirW();
    bases.push_back(exeDir);
    bases.push_back(exeDir + L"\\..\\..");
    bases.push_back(GetCwdW());

#ifdef __FILEW__
    std::wstring src = std::wstring(L"") + __FILEW__;
    ToBackslashes(src);
    size_t pos = src.find_last_of(L"\\/");
    if (pos != std::wstring::npos) {
        std::wstring srcDir = src.substr(0, pos); // ...\engine\gfx
        bases.push_back(srcDir);
        bases.push_back(srcDir + L"\\..\\..");
    }
#endif

    // Try exact path first, then the default shader location.
    std::vector<std::wstring> tried;
    for (const auto& b : bases) {
        std::wstring p1 = b + L"\\" + rel;
        std::wstring p2 = b + L"\\engine\\gfx\\d3d12_shaders\\points.hlsl";
        tried.push_back(p1);
        tried.push_back(p2);
        if (FileExistsW(p1)) return p1;
        if (FileExistsW(p2)) return p2;
    }

    // Log attempted paths to aid debugging and return best guess.
    OutputDebugStringA("[HLSL] ResolveShaderPath failed, tried:\n");
    for (auto& t : tried) {
        OutputDebugStringW((t + L"\n").c_str());
    }
    return exeDir + L"\\" + rel;
}

// Create a root signature with constants + two SRV tables (t0,t1).
// Try RS 1.1, fall back to RS 1.0 when needed.
static Microsoft::WRL::ComPtr<ID3D12RootSignature> CreateRootSignatureGfx(ID3D12Device* dev) {
    using Microsoft::WRL::ComPtr;

    // Descriptor ranges: particles(t0) + palette/velocity(t1).
    D3D12_DESCRIPTOR_RANGE1 rangeParticles{};
    rangeParticles.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    rangeParticles.NumDescriptors = 1;
    rangeParticles.BaseShaderRegister = 0; // t0
    rangeParticles.RegisterSpace = 0;
    rangeParticles.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_NONE;
    rangeParticles.OffsetInDescriptorsFromTableStart = 0;

    D3D12_DESCRIPTOR_RANGE1 rangePalette{};
    rangePalette.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    rangePalette.NumDescriptors = 1;
    rangePalette.BaseShaderRegister = 1; // t1
    rangePalette.RegisterSpace = 0;
    rangePalette.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_NONE;
    rangePalette.OffsetInDescriptorsFromTableStart = 0;

    // Parameters: b0 constants + t0 particles + t1 palette/velocity.
    D3D12_ROOT_PARAMETER1 params[3] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[0].Constants.ShaderRegister = 0;
    params[0].Constants.RegisterSpace = 0;
    params[0].Constants.Num32BitValues = sizeof(PerFrameCB) / 4;

    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_DESCRIPTOR_TABLE1 tbl0{};
    tbl0.NumDescriptorRanges = 1;
    tbl0.pDescriptorRanges = &rangeParticles;
    params[1].DescriptorTable = tbl0;

    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_DESCRIPTOR_TABLE1 tbl1{};
    tbl1.NumDescriptorRanges = 1;
    tbl1.pDescriptorRanges = &rangePalette;
    params[2].DescriptorTable = tbl1;

    Microsoft::WRL::ComPtr<ID3DBlob> blob;
    Microsoft::WRL::ComPtr<ID3DBlob> err;

    // Prefer RS 1.1.
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC vdesc{};
    vdesc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    vdesc.Desc_1_1.NumParameters = _countof(params);
    vdesc.Desc_1_1.pParameters = params;
    vdesc.Desc_1_1.NumStaticSamplers = 0;
    vdesc.Desc_1_1.pStaticSamplers = nullptr;
    vdesc.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

    HRESULT hr = D3D12SerializeVersionedRootSignature(&vdesc, &blob, &err);
    if (FAILED(hr)) {
        // Log and fall back to RS 1.0.
        if (err) {
            OutputDebugStringA("[RS] SerializeVersionedRootSignature(1.1) failed, fallback to 1.0.\n");
            OutputDebugStringA((char*)err->GetBufferPointer());
        }

        D3D12_ROOT_PARAMETER rootParams[4]{};

        rootParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        rootParams[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        rootParams[0].Constants.ShaderRegister = 0;
        rootParams[0].Constants.RegisterSpace = 0;
        rootParams[0].Constants.Num32BitValues = sizeof(PerFrameCB) / 4;

        rootParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParams[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        D3D12_DESCRIPTOR_RANGE range0{};
        range0.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        range0.NumDescriptors = 1;
        range0.BaseShaderRegister = 0;
        range0.RegisterSpace = 0;
        range0.OffsetInDescriptorsFromTableStart = 0;
        rootParams[1].DescriptorTable.NumDescriptorRanges = 1;
        rootParams[1].DescriptorTable.pDescriptorRanges = &range0;

        rootParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParams[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        D3D12_DESCRIPTOR_RANGE range1{};
        range1.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        range1.NumDescriptors = 1;
        range1.BaseShaderRegister = 1;
        range1.RegisterSpace = 0;
        range1.OffsetInDescriptorsFromTableStart = 0;
        rootParams[2].DescriptorTable.NumDescriptorRanges = 1;
        rootParams[2].DescriptorTable.pDescriptorRanges = &range1;

        rootParams[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParams[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        D3D12_DESCRIPTOR_RANGE range2{};
        range2.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        range2.NumDescriptors = 1;
        range2.BaseShaderRegister = 2;
        range2.RegisterSpace = 0;
        range2.OffsetInDescriptorsFromTableStart = 0;
        rootParams[3].DescriptorTable.NumDescriptorRanges = 1;
        rootParams[3].DescriptorTable.pDescriptorRanges = &range2;

        D3D12_ROOT_SIGNATURE_DESC desc10{};
        desc10.NumParameters = _countof(rootParams);
        desc10.pParameters = rootParams;
        desc10.NumStaticSamplers = 0;
        desc10.pStaticSamplers = nullptr;
        desc10.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

        err.Reset();
        hr = D3D12SerializeRootSignature(&desc10, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err);
        if (FAILED(hr)) {
            if (err) OutputDebugStringA((char*)err->GetBufferPointer());
            return nullptr;
        }
    }

    Microsoft::WRL::ComPtr<ID3D12RootSignature> rs;
    if (FAILED(dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&rs)))) {
        return nullptr;
    }
    return rs;
}

// Simple alpha blend state for rendering points/quads.
static D3D12_BLEND_DESC AlphaBlendDesc() {
    D3D12_BLEND_DESC b{};
    b.AlphaToCoverageEnable = FALSE;
    b.IndependentBlendEnable = FALSE;

    auto& rt = b.RenderTarget[0];
    rt.BlendEnable = TRUE;
    rt.SrcBlend = D3D12_BLEND_SRC_ALPHA;
    rt.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
    rt.BlendOp = D3D12_BLEND_OP_ADD;
    rt.SrcBlendAlpha = D3D12_BLEND_ONE;
    rt.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
    rt.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    return b;
}

// No-cull rasterizer state for billboards.
static D3D12_RASTERIZER_DESC DefaultRasterizerNoCull() {
    D3D12_RASTERIZER_DESC r{};
    r.FillMode = D3D12_FILL_MODE_SOLID;
    r.CullMode = D3D12_CULL_MODE_NONE;
    r.FrontCounterClockwise = FALSE;
    r.DepthBias = D3D12_DEFAULT_DEPTH_BIAS;
    r.DepthBiasClamp = D3D12_DEFAULT_DEPTH_BIAS_CLAMP;
    r.SlopeScaledDepthBias = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS;
    r.DepthClipEnable = TRUE;
    r.MultisampleEnable = FALSE;
    r.AntialiasedLineEnable = FALSE;
    r.ForcedSampleCount = 0;
    r.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
    return r;
}

// Compile HLSL with manual file IO to strip BOM / leading control chars.
static Microsoft::WRL::ComPtr<ID3DBlob> Compile(const wchar_t* relativePath, const char* entry, const char* target) {
    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#if defined(_DEBUG)
    flags |= D3DCOMPILE_DEBUG;
#endif

    std::wstring full = ResolveShaderPath(relativePath);

    // Read source bytes and scrub BOM/invalid prefixes.
    std::ifstream ifs(full, std::ios::binary);
    if (!ifs.is_open()) {
        OutputDebugStringW((L"[HLSL] Open failed: " + full + L"\n").c_str());
        return nullptr;
    }
    std::vector<char> bytes((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

    if (bytes.empty()) {
        OutputDebugStringW((L"[HLSL] Empty file: " + full + L"\n").c_str());
        return nullptr;
    }

    // Remove UTF-8 BOM.
    if (bytes.size() >= 3 &&
        (unsigned char)bytes[0] == 0xEF &&
        (unsigned char)bytes[1] == 0xBB &&
        (unsigned char)bytes[2] == 0xBF) {
        bytes.erase(bytes.begin(), bytes.begin() + 3);
    }

    // Strip first non-printable control characters.
    while (!bytes.empty()) {
        unsigned char c = (unsigned char)bytes[0];
        if (c == 0x09 || c == 0x0A || c == 0x0D || (c >= 0x20 && c < 0x7F)) break;
        bytes.erase(bytes.begin());
    }
    if (bytes.empty()) {
        OutputDebugStringW((L"[HLSL] All leading bytes stripped, nothing left: " + full + L"\n").c_str());
        return nullptr;
    }

    Microsoft::WRL::ComPtr<ID3DBlob> cs, err;
    HRESULT hr = D3DCompile(
        bytes.data(),
        bytes.size(),
        /*sourceName*/ nullptr,
        /*defines*/ nullptr,
        D3D_COMPILE_STANDARD_FILE_INCLUDE,
        entry,
        target,
        flags,
        0,
        &cs,
        &err);

    if (FAILED(hr)) {
        if (err) OutputDebugStringA((char*)err->GetBufferPointer());
        std::wstring msg = L"[HLSL] Compile failed: " + full + L"\n";
        OutputDebugStringW(msg.c_str());

        // Dump first bytes to debug weird encodings.
        char hexBuf[256];
        int pos = 0;
        pos += snprintf(hexBuf + pos, sizeof(hexBuf) - pos, "[HLSL] First bytes:");
        for (size_t i = 0; i < std::min<size_t>(16, bytes.size()); ++i) {
            pos += snprintf(hexBuf + pos, sizeof(hexBuf) - pos, " %02X", (unsigned char)bytes[i]);
        }
        pos += snprintf(hexBuf + pos, sizeof(hexBuf) - pos, "\n");
        OutputDebugStringA(hexBuf);
        return nullptr;
    }

    std::wstring ok = L"[HLSL] Compile ok: " + full + L"\n";
    OutputDebugStringW(ok.c_str());
    return cs;
}

} // namespace

// Initialize device, SRV heap, shared fence, and build frame graph.
bool RendererD3D12::Initialize(HWND hwnd, const RenderInitParams& p) {
    prof::Range r("Renderer.Initialize", prof::Color(0x20, 0x80, 0xC0));

    DeviceInitParams dp;
    dp.width = p.width;
    dp.height = p.height;
    dp.bufferCount = 3;
    dp.vsync = p.vsync;

    if (!m_device.initialize(hwnd, dp)) return false;
    m_device.createSrvHeap(256, true);

    std::memcpy(m_clearColor, m_visual.clearColor, sizeof(m_clearColor));

    // Shared timeline fence for interop.
    if (FAILED(m_device.device()->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&m_timelineFence)))) return false;
    if (FAILED(m_device.device()->CreateSharedHandle(m_timelineFence.Get(), nullptr, GENERIC_ALL, nullptr, &m_timelineFenceSharedHandle))) return false;

    m_renderFenceValue = 0;
    m_lastSimFenceValue = 0;

    BuildFrameGraph();
    g_unityExportRenderer = this;
    return true;
}

// Release resources and cached shared handles.
void RendererD3D12::Shutdown() {
    m_sharedParticleBuffer.Reset();
    m_paletteBuffer.Reset();
    m_sharedVelocityBuffer.Reset();

    for (int i = 0; i < 2; ++i) {
        if (g_cachedSharedParticleHandles[i]) {
            CloseHandle(g_cachedSharedParticleHandles[i]);
            g_cachedSharedParticleHandles[i] = nullptr;
        }
    }

    if (m_timelineFenceSharedHandle) {
        CloseHandle(m_timelineFenceSharedHandle);
        m_timelineFenceSharedHandle = nullptr;
    }

    m_timelineFence.Reset();
    m_device.shutdown();

    if (g_unityExportRenderer == this) g_unityExportRenderer = nullptr;
}

// Queue-side wait on simulation fence; track last waited value.
void RendererD3D12::WaitSimulationFence(uint64_t simValue) {
    if (!m_timelineFence) return;
    if (simValue > m_lastSimFenceValue) {
        m_device.queue()->Wait(m_timelineFence.Get(), simValue);
        m_lastSimFenceValue = simValue;
    }
}

// Signal render completion so simulation can optionally wait.
void RendererD3D12::SignalRenderComplete(uint64_t /*lastSimValue*/) {
    if (!m_timelineFence) return;
    ++m_renderFenceValue;
    m_device.queue()->Signal(m_timelineFence.Get(), m_renderFenceValue);
}

// Create a single shared particle buffer and SRV, return shared handle.
bool RendererD3D12::CreateSharedParticleBuffer(uint32_t numElements, uint32_t strideBytes, HANDLE& outSharedHandle) {
    outSharedHandle = nullptr;

    const UINT64 sizeBytes = UINT64(numElements) * UINT64(strideBytes);

    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = sizeBytes;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc = { 1, 0 };
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = D3D12_RESOURCE_FLAG_NONE;

    Microsoft::WRL::ComPtr<ID3D12Resource> res;
    if (FAILED(m_device.device()->CreateCommittedResource(
            &hp,
            D3D12_HEAP_FLAG_SHARED,
            &rd,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&res)))) {
        return false;
    }

    int srvIndex = m_device.createBufferSRV(res.Get(), numElements, strideBytes);
    if (srvIndex < 0) return false;

    HANDLE handle = nullptr;
    if (FAILED(m_device.device()->CreateSharedHandle(res.Get(), nullptr, GENERIC_ALL, nullptr, &handle))) return false;

    m_sharedParticleBuffer = res;
    m_particleSrvIndex = srvIndex;
    m_particleCount = numElements;
    outSharedHandle = handle;
    return true;
}

// Create indexed ping-pong shared buffers; SRVs recorded per slot.
bool RendererD3D12::CreateSharedParticleBufferIndexed(int slot, uint32_t numElements, uint32_t strideBytes, HANDLE& outSharedHandle) {
    outSharedHandle = nullptr;
    if (slot < 0 || slot > 1) return false;

    const UINT64 sizeBytes = UINT64(numElements) * UINT64(strideBytes);

    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = sizeBytes;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc = { 1, 0 };
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    Microsoft::WRL::ComPtr<ID3D12Resource> res;
    if (FAILED(m_device.device()->CreateCommittedResource(
            &hp,
            D3D12_HEAP_FLAG_SHARED,
            &rd,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&res)))) {
        return false;
    }

    int srvIndex = m_device.createBufferSRV(res.Get(), numElements, strideBytes);
    if (srvIndex < 0) return false;

    HANDLE handle = nullptr;
    if (FAILED(m_device.device()->CreateSharedHandle(res.Get(), nullptr, GENERIC_ALL, nullptr, &handle))) return false;

    m_sharedParticleBuffers[slot] = res;
    m_particleSrvIndexPing[slot] = srvIndex;

    if (slot == 0) {
        m_activePingIndex = 0;
        if (m_particleSrvIndex < 0) {
            m_particleSrvIndex = srvIndex; // Defensive default.
        }
    }

    if (numElements > m_particleCount) m_particleCount = numElements;

    outSharedHandle = handle;
    m_sharedParticleBufferHandles[slot] = handle;
    return true;
}

// Create shared velocity buffer and SRV.
bool RendererD3D12::CreateSharedVelocityBuffer(uint32_t numElements, uint32_t strideBytes, HANDLE& outSharedHandle) {
    outSharedHandle = nullptr;
    if (numElements == 0 || strideBytes == 0) return false;

    const UINT64 sizeBytes = UINT64(numElements) * UINT64(strideBytes);

    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = sizeBytes;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc = { 1, 0 };
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    Microsoft::WRL::ComPtr<ID3D12Resource> res;
    if (FAILED(m_device.device()->CreateCommittedResource(
            &hp,
            D3D12_HEAP_FLAG_SHARED,
            &rd,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(&res)))) {
        return false;
    }

    int srvIndex = m_device.createBufferSRV(res.Get(), numElements, strideBytes);
    if (srvIndex < 0) return false;

    HANDLE handle = nullptr;
    if (FAILED(m_device.device()->CreateSharedHandle(res.Get(), nullptr, GENERIC_ALL, nullptr, &handle))) return false;

    m_sharedVelocityBuffer = res;
    m_velocitySrvIndex = srvIndex;
    outSharedHandle = handle;
    return true;
}

// Register known CUDA pointers for ping-pong SRV selection.
void RendererD3D12::RegisterPingPongCudaPtrs(const void* ptrA, const void* ptrB) {
    m_knownCudaPtrs[0] = const_cast<void*>(ptrA);
    m_knownCudaPtrs[1] = const_cast<void*>(ptrB);

    if (ptrA) {
        UpdateParticleSRVForPingPong(ptrA);
    }

    std::fprintf(
        stderr,
        "[Render.PingPong][Register] ptrA=%p ptrB=%p activePing=%d srvIndex=%d srvA=%d srvB=%d\n",
        ptrA,
        ptrB,
        m_activePingIndex,
        m_particleSrvIndex,
        m_particleSrvIndexPing[0],
        m_particleSrvIndexPing[1]);
}

// Select active SRV based on current CUDA device pointer.
void RendererD3D12::UpdateParticleSRVForPingPong(const void* devicePtrCurr) {
    if (!devicePtrCurr) return;

    for (int i = 0; i < 2; ++i) {
        if (m_knownCudaPtrs[i] == devicePtrCurr && m_particleSrvIndexPing[i] >= 0) {
            m_activePingIndex = i;
            m_particleSrvIndex = m_particleSrvIndexPing[i];
            return;
        }
    }
    // No match: keep existing SRV (single-buffer mode).
}

// Open shared velocity resource and create SRV for speed coloring.
bool RendererD3D12::ImportSharedVelocityAsSRV(HANDLE sharedHandle, uint32_t numElements, uint32_t strideBytes, int& outSrvIndex) {
    outSrvIndex = -1;

    ID3D12Resource* res = nullptr;
    if (!m_device.openSharedResource(sharedHandle, __uuidof(ID3D12Resource), (void**)&res)) return false;

    Microsoft::WRL::ComPtr<ID3D12Resource> local;
    local.Attach(res);

    int srvIndex = m_device.createBufferSRV(local.Get(), numElements, strideBytes);
    if (srvIndex < 0) return false;

    m_sharedVelocityBuffer = local;
    m_velocitySrvIndex = srvIndex;
    outSrvIndex = srvIndex;
    return true;
}

// Build frame graph: clear -> points/group -> points/speed -> present.
void RendererD3D12::BuildFrameGraph() {
    prof::Range r("Renderer.BuildFrameGraph", prof::Color(0x90, 0x40, 0x30));

    m_fg = core::FrameGraph{};

    auto rsGfx = CreateRootSignatureGfx(m_device.device());

    // Compile group-palette (float) shader variants.
    auto vsFloat = Compile(L"engine\\gfx\\d3d12_shaders\\points.hlsl", "VSMain", "vs_5_1");
    auto psFloat = Compile(L"engine\\gfx\\d3d12_shaders\\points.hlsl", "PSMain", "ps_5_1");

    // Compile speed-colored shader variants (optional).
    auto vsSpeed = Compile(L"engine\\gfx\\d3d12_shaders\\points_speed.hlsl", "VSMain", "vs_5_1");
    auto psSpeed = Compile(L"engine\\gfx\\d3d12_shaders\\points_speed.hlsl", "PSMain", "ps_5_1");

    // PSO: group/palette path.
    m_psoPointsFloat.Reset();
    if (rsGfx && vsFloat && psFloat) {
        DXGI_FORMAT rtFmt = m_device.currentBackbuffer()->GetDesc().Format;

        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = rsGfx.Get();
        pso.VS = { vsFloat->GetBufferPointer(), vsFloat->GetBufferSize() };
        pso.PS = { psFloat->GetBufferPointer(), psFloat->GetBufferSize() };
        pso.BlendState = AlphaBlendDesc();
        pso.SampleMask = UINT_MAX;
        pso.RasterizerState = DefaultRasterizerNoCull();
        pso.DepthStencilState.DepthEnable = FALSE;
        pso.DepthStencilState.StencilEnable = FALSE;
        pso.InputLayout = { nullptr, 0 };
        pso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pso.NumRenderTargets = 1;
        pso.RTVFormats[0] = rtFmt;
        pso.SampleDesc.Count = 1;

        if (SUCCEEDED(m_device.device()->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(&m_psoPointsFloat)))) {
            OutputDebugStringA("[PSO] Created float points pipeline.\n");
        }
    }

    // PSO: speed-color path (position + velocity).
    m_psoPointsSpeed.Reset();
    if (rsGfx && vsSpeed && psSpeed) {
        DXGI_FORMAT rtFmt = m_device.currentBackbuffer()->GetDesc().Format;

        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = rsGfx.Get();
        pso.VS = { vsSpeed->GetBufferPointer(), vsSpeed->GetBufferSize() };
        pso.PS = { psSpeed->GetBufferPointer(), psSpeed->GetBufferSize() };
        pso.BlendState = AlphaBlendDesc();
        pso.SampleMask = UINT_MAX;
        pso.RasterizerState = DefaultRasterizerNoCull();
        pso.DepthStencilState.DepthEnable = FALSE;
        pso.DepthStencilState.StencilEnable = FALSE;
        pso.InputLayout = { nullptr, 0 };
        pso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pso.NumRenderTargets = 1;
        pso.RTVFormats[0] = rtFmt;
        pso.SampleDesc.Count = 1;

        if (SUCCEEDED(m_device.device()->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(&m_psoPointsSpeed)))) {
            OutputDebugStringA("[PSO] Created speed points pipeline.\n");
        }
    }

    // Pass: clear backbuffer.
    core::PassDesc clear{};
    clear.name = "clear";
    clear.execute = [this]() {
        m_device.beginFrame();
        std::memcpy(m_clearColor, m_visual.clearColor, sizeof(m_clearColor));
        m_device.clearCurrentRTV(m_clearColor);
        m_device.writeTimestamp();
    };
    m_fg.addPass(clear);

    // Pass: group-palette draw (uses palette SRV at t1).
    if (rsGfx && m_psoPointsFloat) {
        core::PassDesc points{};
        points.name = "points";
        points.execute = [this, rsGfx]() {
            if (m_renderMode != RendererD3D12::RenderMode::GroupPalette) return;

            auto* cl = m_device.cmdList();
            auto rtvHandle = m_device.currentRTV();

            cl->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);

            D3D12_VIEWPORT vp{ 0.f, 0.f, (float)m_device.width(), (float)m_device.height(), 0.f, 1.f };
            D3D12_RECT sc{ 0, 0, (LONG)m_device.width(), (LONG)m_device.height() };
            cl->RSSetViewports(1, &vp);
            cl->RSSetScissorRects(1, &sc);

            ID3D12DescriptorHeap* heaps[] = { m_device.srvHeap() };
            cl->SetDescriptorHeaps(1, heaps);
            cl->SetGraphicsRootSignature(rsGfx.Get());
            cl->SetPipelineState(m_psoPointsFloat.Get());
            cl->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

            // Transition particle buffer to read.
            if (m_sharedParticleBuffer) {
                D3D12_RESOURCE_BARRIER b{};
                b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                b.Transition.pResource = m_sharedParticleBuffer.Get();
                b.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
                b.Transition.StateAfter = D3D12_RESOURCE_STATE_GENERIC_READ;
                b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                cl->ResourceBarrier(1, &b);
            }

            // Constants (view-proj + screen/visual settings + grouping).
            PerFrameCB cb{};
            float3 eye = m_camera.eye, at = m_camera.at, up = m_camera.up;
            Mat4 V = lookAtRH(eye, at, up);
            float aspect = (m_device.height() > 0) ? (float)m_device.width() / (float)m_device.height() : 1.0f;
            Mat4 P = perspectiveFovRH_ZO(Deg2Rad(m_camera.fovYDeg), aspect, m_camera.nearZ, m_camera.farZ);
            Mat4 VP = mul(P, V);
            std::memcpy(cb.viewProj, VP.m, sizeof(cb.viewProj));
            cb.screenSize[0] = (float)m_device.width();
            cb.screenSize[1] = (float)m_device.height();
            cb.particleRadiusPx = m_visual.particleRadiusPx;
            cb.thicknessScale = m_visual.thicknessScale;
            cb.groups = m_groups;
            cb.particlesPerGroup = m_particlesPerGroup;
            cl->SetGraphicsRoot32BitConstants(0, sizeof(PerFrameCB) / 4, &cb, 0);

            // Bind SRVs: position (t0) + palette (t1).
            if (m_particleSrvIndex >= 0 && m_particleCount > 0) {
                auto gpuPos = m_device.srvGpuHandleAt((uint32_t)m_particleSrvIndex);
                cl->SetGraphicsRootDescriptorTable(1, gpuPos);
            }
            if (m_paletteSrvIndex >= 0) {
                auto gpuPal = m_device.srvGpuHandleAt((uint32_t)m_paletteSrvIndex);
                cl->SetGraphicsRootDescriptorTable(2, gpuPal);
            }

            // Draw N instances (each expands to quad in shader).
            UINT instanceCount = m_particleCount;
            if (instanceCount > 0) {
                cl->DrawInstanced(6, instanceCount, 0, 0);
            }

            // Restore resource state to COMMON.
            if (m_sharedParticleBuffer) {
                D3D12_RESOURCE_BARRIER b{};
                b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                b.Transition.pResource = m_sharedParticleBuffer.Get();
                b.Transition.StateBefore = D3D12_RESOURCE_STATE_GENERIC_READ;
                b.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
                b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                cl->ResourceBarrier(1, &b);
            }

            m_device.writeTimestamp();
        };
        m_fg.addPass(points);
    }

    // Pass: speed-colored draw (position t0 + velocity t1).
    if (rsGfx && m_psoPointsSpeed) {
        core::PassDesc pointsSpeed{};
        pointsSpeed.name = "points_speed";
        pointsSpeed.execute = [this, rsGfx]() {
            if (m_renderMode != RendererD3D12::RenderMode::SpeedColor) return;

            // Validate availability; log for diagnostics.
            if (m_particleCount == 0 || m_particleSrvIndex < 0) {
                char dbg[512];
                _snprintf_s(
                    dbg,
                    sizeof(dbg),
                    _TRUNCATE,
                    "[SpeedPass] SKIP draw: count=%u srvPos=%d srvVel=%d activePing=%d shared=%p ping0=%p ping1=%p\n",
                    (unsigned)m_particleCount,
                    (int)m_particleSrvIndex,
                    (int)m_velocitySrvIndex,
                    (int)m_activePingIndex,
                    m_sharedParticleBuffer.Get(),
                    m_sharedParticleBuffers[0].Get(),
                    m_sharedParticleBuffers[1].Get());
                OutputDebugStringA(dbg);
                return;
            }

            auto* cl = m_device.cmdList();
            auto rtvHandle = m_device.currentRTV();

            cl->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);

            D3D12_VIEWPORT vp{ 0.f, 0.f, (float)m_device.width(), (float)m_device.height(), 0.f, 1.f };
            D3D12_RECT sc{ 0, 0, (LONG)m_device.width(), (LONG)m_device.height() };
            cl->RSSetViewports(1, &vp);
            cl->RSSetScissorRects(1, &sc);

            ID3D12DescriptorHeap* heaps[] = { m_device.srvHeap() };
            cl->SetDescriptorHeaps(1, heaps);
            cl->SetGraphicsRootSignature(rsGfx.Get());
            cl->SetPipelineState(m_psoPointsSpeed.Get());
            cl->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

            // Transition particles (single or ping) to read.
            if (m_sharedParticleBuffer) {
                D3D12_RESOURCE_BARRIER b{};
                b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                b.Transition.pResource = m_sharedParticleBuffer.Get();
                b.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
                b.Transition.StateAfter = D3D12_RESOURCE_STATE_GENERIC_READ;
                b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                cl->ResourceBarrier(1, &b);
            } else {
                if (m_activePingIndex >= 0 && m_activePingIndex < 2) {
                    auto& pingRes = m_sharedParticleBuffers[m_activePingIndex];
                    if (pingRes) {
                        D3D12_RESOURCE_BARRIER b{};
                        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                        b.Transition.pResource = pingRes.Get();
                        b.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
                        b.Transition.StateAfter = D3D12_RESOURCE_STATE_GENERIC_READ;
                        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                        cl->ResourceBarrier(1, &b);
                    }
                }
            }

            // Transition velocity to read (if available).
            if (m_sharedVelocityBuffer) {
                D3D12_RESOURCE_BARRIER b{};
                b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                b.Transition.pResource = m_sharedVelocityBuffer.Get();
                b.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
                b.Transition.StateAfter = D3D12_RESOURCE_STATE_GENERIC_READ;
                b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                cl->ResourceBarrier(1, &b);
            }

            // Constants.
            PerFrameCB cb{};
            float3 eye = m_camera.eye, at = m_camera.at, up = m_camera.up;
            Mat4 V = lookAtRH(eye, at, up);
            float aspect = (m_device.height() > 0) ? (float)m_device.width() / (float)m_device.height() : 1.0f;
            Mat4 P = perspectiveFovRH_ZO(Deg2Rad(m_camera.fovYDeg), aspect, m_camera.nearZ, m_camera.farZ);
            Mat4 VP = mul(P, V);
            std::memcpy(cb.viewProj, VP.m, sizeof(cb.viewProj));
            cb.screenSize[0] = (float)m_device.width();
            cb.screenSize[1] = (float)m_device.height();
            cb.particleRadiusPx = m_visual.particleRadiusPx;
            cb.thicknessScale = m_visual.thicknessScale;
            cb.groups = m_groups;
            cb.particlesPerGroup = m_particlesPerGroup;
            cl->SetGraphicsRoot32BitConstants(0, sizeof(PerFrameCB) / 4, &cb, 0);

            // Bind position SRV (t0).
            auto gpuPos = m_device.srvGpuHandleAt((uint32_t)m_particleSrvIndex);
            cl->SetGraphicsRootDescriptorTable(1, gpuPos);

            // Resolve t1 source: velocity -> palette -> position (fallbacks).
            int srvVel = -1;
            if (m_velocitySrvIndex >= 0) {
                srvVel = m_velocitySrvIndex;
            } else if (m_paletteSrvIndex >= 0) {
                srvVel = m_paletteSrvIndex;
                OutputDebugStringA("[SpeedPass] velocity SRV missing, using palette SRV as fallback.\n");
            } else {
                srvVel = m_particleSrvIndex;
                OutputDebugStringA("[SpeedPass] velocity & palette SRV missing, reusing position SRV as fallback.\n");
            }
            if (srvVel >= 0) {
                auto gpuVel = m_device.srvGpuHandleAt((uint32_t)srvVel);
                cl->SetGraphicsRootDescriptorTable(2, gpuVel);
            }

            // Draw instances.
            UINT instanceCount = m_particleCount;
            if (instanceCount > 0) {
                cl->DrawInstanced(6, instanceCount, 0, 0);
            }

            // Restore states after draw.
            auto barrierBack = [&](Microsoft::WRL::ComPtr<ID3D12Resource>& res) {
                if (!res) return;
                D3D12_RESOURCE_BARRIER b{};
                b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                b.Transition.pResource = res.Get();
                b.Transition.StateBefore = D3D12_RESOURCE_STATE_GENERIC_READ;
                b.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
                b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                cl->ResourceBarrier(1, &b);
            };

            if (m_sharedParticleBuffer) {
                barrierBack(m_sharedParticleBuffer);
            } else if (m_activePingIndex >= 0 && m_activePingIndex < 2) {
                barrierBack(m_sharedParticleBuffers[m_activePingIndex]);
            }
            if (m_sharedVelocityBuffer) {
                barrierBack(m_sharedVelocityBuffer);
            }

            m_device.writeTimestamp();
        };
        m_fg.addPass(pointsSpeed);
    }

    // Pass: present backbuffer.
    core::PassDesc present{};
    present.name = "present";
    present.execute = [this]() { m_device.present(); };
    m_fg.addPass(present);

    m_fg.compile();
}

// Per-frame rendering: execute frame graph and record GPU times.
void RendererD3D12::RenderFrame(core::Profiler& profiler) {
    prof::Range r("Renderer.RenderFrame", prof::Color(0x40, 0x90, 0x50));

    static uint64_t s_debugFrame = 0;
    if (s_debugFrame < 8 || (s_debugFrame % 120) == 0) {
        char dbg[256];
        _snprintf_s(
            dbg,
            sizeof(dbg),
            _TRUNCATE,
            "[Renderer] frame=%llu mode=%d count=%u srvPos=%d srvVel=%d activePing=%d\n",
            (unsigned long long)s_debugFrame,
            (int)m_renderMode,
            (unsigned)m_particleCount,
            (int)m_particleSrvIndex,
            (int)m_velocitySrvIndex,
            (int)m_activePingIndex);
        OutputDebugStringA(dbg);
    }
    ++s_debugFrame;

    std::vector<double> gpuMs;
    m_fg.execute([&](const std::string& name, double ms) { profiler.addRow(name, ms); });
    if (m_device.readbackPassTimesMs(gpuMs)) {
        for (size_t i = 0; i < gpuMs.size(); ++i) {
            profiler.addRow(std::string("gpu_") + std::to_string(i), gpuMs[i]);
        }
    }
}

// Import a shared buffer as particle SRV (single-buffer flow).
bool RendererD3D12::ImportSharedBufferAsSRV(HANDLE sharedHandle, uint32_t numElements, uint32_t strideBytes, int& outSrvIndex) {
    ID3D12Resource* res = nullptr;
    if (!m_device.openSharedResource(sharedHandle, __uuidof(ID3D12Resource), (void**)&res)) return false;

    Microsoft::WRL::ComPtr<ID3D12Resource> local;
    local.Attach(res);

    int srvIndex = m_device.createBufferSRV(local.Get(), numElements, strideBytes);
    if (srvIndex < 0) return false;

    m_sharedParticleBuffer = local;
    m_particleSrvIndex = srvIndex;
    m_particleCount = numElements;
    outSrvIndex = srvIndex;
    m_useHalfRender = false;
    return true;
}

// Full GPU wait (CPU-side) for frame completion when needed.
void RendererD3D12::WaitForGPU() {
    prof::Range r("Renderer.WaitForGPU", prof::Color(0xAA, 0x20, 0x20));
    m_device.waitForGPU();
}

// Update group palette buffer and SRV (upload heap for simplicity).
bool RendererD3D12::UpdateGroupPalette(const float* rgbTriples, uint32_t groupCount) {
    if (!rgbTriples || groupCount == 0) {
        m_groups = 0;
        m_paletteSrvIndex = -1;
        m_paletteBuffer.Reset();
        return true;
    }

    size_t bytes = sizeof(float) * 3 * groupCount;

    D3D12_HEAP_PROPERTIES hpUp{};
    hpUp.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = bytes;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc = { 1, 0 };
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = D3D12_RESOURCE_FLAG_NONE;

    Microsoft::WRL::ComPtr<ID3D12Resource> buf;
    if (FAILED(m_device.device()->CreateCommittedResource(
            &hpUp,
            D3D12_HEAP_FLAG_NONE,
            &rd,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&buf)))) {
        return false;
    }

    void* mapped = nullptr;
    D3D12_RANGE range{ 0, 0 };
    if (FAILED(buf->Map(0, &range, &mapped))) return false;

    std::memcpy(mapped, rgbTriples, bytes);
    buf->Unmap(0, nullptr);

    m_paletteBuffer = buf;
    m_paletteSrvIndex = m_device.createBufferSRV(m_paletteBuffer.Get(), groupCount, sizeof(float) * 3);
    return (m_paletteSrvIndex >= 0);
}

} // namespace gfx
