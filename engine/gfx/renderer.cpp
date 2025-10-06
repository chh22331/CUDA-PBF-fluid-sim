#include "renderer.h"
#include "../../sim/cuda_vec_math.cuh"
#include <chrono>
#include <d3dcompiler.h>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#pragma comment(lib, "d3dcompiler.lib")

namespace gfx {

    namespace {

        struct Mat4 { float m[16]; };

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
            float3 x = make_float3(up.y * z.z - up.z * z.y,
                up.z * z.x - up.x * z.z,
                up.x * z.y - up.y * z.x);
            float lenx = sqrtf(x.x * x.x + x.y * x.y + x.z * x.z);
            x.x /= lenx; x.y /= lenx; x.z /= lenx;
            float3 y = make_float3(z.y * x.z - z.z * x.y,
                z.z * x.x - z.x * x.z,
                z.x * x.y - z.y * x.x);
            Mat4 m{};
            m.m[0] = x.x; m.m[4] = x.y; m.m[8] = x.z;  m.m[12] = -(x.x * eye.x + x.y * eye.y + x.z * eye.z);
            m.m[1] = y.x; m.m[5] = y.y; m.m[9] = y.z;  m.m[13] = -(y.x * eye.x + y.y * eye.y + y.z * eye.z);
            m.m[2] = z.x; m.m[6] = z.y; m.m[10] = z.z;  m.m[14] = -(z.x * eye.x + z.y * eye.y + z.z * eye.z);
            m.m[3] = 0.f; m.m[7] = 0.f; m.m[11] = 0.f;  m.m[15] = 1.f;
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

        static float Deg2Rad(float deg) { return deg * 3.14159265358979323846f / 180.0f; }

        // 路径工具
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
            std::wstring out(n, L'\0');
            if (n > 0) {
                GetCurrentDirectoryW(n, out.data());
                if (!out.empty() && out.back() == L'\0') out.pop_back();
            }
            return out;
        }

        static bool FileExistsW(const std::wstring& p) {
            DWORD a = GetFileAttributesW(p.c_str());
            return (a != INVALID_FILE_ATTRIBUTES) && !(a & FILE_ATTRIBUTE_DIRECTORY);
        }

        static void ToBackslashes(std::wstring& s) { for (auto& ch : s) if (ch == L'/') ch = L'\\'; }

        // 在常见位置搜索着色器文件，返回第一个存在的绝对路径
        static std::wstring ResolveShaderPath(const wchar_t* relative) {
            std::wstring rel = relative ? std::wstring(relative) : std::wstring();
            ToBackslashes(rel);

            // 已绝对路径
            if (!rel.empty() && (rel[0] == L'\\' || (rel.size() > 1 && rel[1] == L':'))) {
                return rel;
            }

            // 候选根目录
            std::vector<std::wstring> bases;
            const std::wstring exeDir = GetExeDirW();
            bases.push_back(exeDir);                           // x64\Release\
                        bases.push_back(exeDir + L"\\..\\..");            // 项目根（从 x64\Release\ 回到 repo 根）
            bases.push_back(GetCwdW());                       // 进程工作目录

#ifdef __FILEW__
            // 以当前源文件目录为根（engine\gfx\），便于开发期运行
            std::wstring src = std::wstring(L"") + __FILEW__;
            ToBackslashes(src);
            size_t pos = src.find_last_of(L"\\/");
            if (pos != std::wstring::npos) {
                std::wstring srcDir = src.substr(0, pos);     // ...\engine\gfx
                bases.push_back(srcDir);                      // 引导到源目录
                bases.push_back(srcDir + L"\\..\\..");        // 源根（repo 根）
            }
#endif

            // 常见的 repo 相对路径：engine\gfx\d3d12_shaders\*.hlsl
            std::vector<std::wstring> tried;
            for (const auto& b : bases) {
                std::wstring p1 = b + L"\\" + rel;                           tried.push_back(p1);
                std::wstring p2 = b + L"\\engine\\gfx\\d3d12_shaders\\points.hlsl"; tried.push_back(p2);
                if (FileExistsW(p1)) return p1;
                if (FileExistsW(p2)) return p2;
            }

            // 失败时把尝试过的路径打到调试输出，便于定位
            OutputDebugStringA("[HLSL] ResolveShaderPath failed, tried:\n");
            for (auto& t : tried) {
                OutputDebugStringW((t + L"\n").c_str());
            }
            // 返回一个最可能的路径（exeDir + rel），让后续编译报错也能显示绝对路径
            return exeDir + L"\\" + rel;
        }

        static Microsoft::WRL::ComPtr<ID3D12RootSignature> CreateRootSignatureGfx(ID3D12Device* dev) {
            using Microsoft::WRL::ComPtr;

            D3D12_DESCRIPTOR_RANGE1 ranges[1] = {};
            ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            ranges[0].NumDescriptors = 1;
            ranges[0].BaseShaderRegister = 0;
            ranges[0].RegisterSpace = 0;
            ranges[0].Flags = D3D12_DESCRIPTOR_RANGE_FLAG_NONE;
            ranges[0].OffsetInDescriptorsFromTableStart = 0;

            D3D12_ROOT_PARAMETER1 params[2] = {};
            params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            params[0].Constants.ShaderRegister = 0; // b0
            params[0].Constants.RegisterSpace = 0;
            params[0].Constants.Num32BitValues = sizeof(PerFrameCB) / 4;

            params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            D3D12_ROOT_DESCRIPTOR_TABLE1 table{}; table.NumDescriptorRanges = 1; table.pDescriptorRanges = ranges;
            params[1].DescriptorTable = table;

            D3D12_VERSIONED_ROOT_SIGNATURE_DESC vdesc{};
            vdesc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
            vdesc.Desc_1_1.NumParameters = _countof(params);
            vdesc.Desc_1_1.pParameters = params;
            vdesc.Desc_1_1.NumStaticSamplers = 0;
            vdesc.Desc_1_1.pStaticSamplers = nullptr;
            vdesc.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

            Microsoft::WRL::ComPtr<ID3DBlob> blob, err;
            if (FAILED(D3D12SerializeVersionedRootSignature(&vdesc, &blob, &err))) {
                if (err) OutputDebugStringA((char*)err->GetBufferPointer());
                return nullptr;
            }
            Microsoft::WRL::ComPtr<ID3D12RootSignature> rs;
            if (FAILED(dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&rs)))) {
                return nullptr;
            }
            return rs;
        }

        D3D12_BLEND_DESC AlphaBlendDesc() {
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

        D3D12_RASTERIZER_DESC DefaultRasterizerNoCull() {
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

        // 编译前先 ResolveShaderPath
        static Microsoft::WRL::ComPtr<ID3DBlob> Compile(const wchar_t* relativePath, const char* entry, const char* target) {
            UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#if defined(_DEBUG)
            flags |= D3DCOMPILE_DEBUG;
#endif
            std::wstring full = ResolveShaderPath(relativePath);
            Microsoft::WRL::ComPtr<ID3DBlob> cs, err;
            HRESULT hr = D3DCompileFromFile(full.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, entry, target, flags, 0, &cs, &err);
            if (FAILED(hr)) {
                if (err) OutputDebugStringA((char*)err->GetBufferPointer());
                std::wstring msg = L"[HLSL] Compile failed: " + full + L"\n";
                OutputDebugStringW(msg.c_str());
                return nullptr;
            }
            std::wstring ok = L"[HLSL] Compile ok: " + full + L"\n";
            OutputDebugStringW(ok.c_str());
            return cs;
        }

    } // namespace

    bool RendererD3D12::Initialize(HWND hwnd, const RenderInitParams& p) {
        DeviceInitParams dp; dp.width = p.width; dp.height = p.height; dp.bufferCount = 3; dp.vsync = p.vsync;
        if (!m_device.initialize(hwnd, dp)) return false;
        m_device.createSrvHeap(256, true);
        std::memcpy(m_clearColor, m_visual.clearColor, sizeof(m_clearColor));
        BuildFrameGraph();
        return true;
    }

    void RendererD3D12::Shutdown() {
        m_sharedParticleBuffer.Reset();
        m_device.shutdown();
    }

    bool RendererD3D12::CreateSharedParticleBuffer(uint32_t numElements, uint32_t strideBytes, HANDLE& outSharedHandle) {
        outSharedHandle = nullptr;

        const UINT64 sizeBytes = UINT64(numElements) * UINT64(strideBytes);

        D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC rd{}; rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; rd.Alignment = 0;
        rd.Width = sizeBytes; rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
        rd.Format = DXGI_FORMAT_UNKNOWN; rd.SampleDesc = { 1, 0 }; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        rd.Flags = D3D12_RESOURCE_FLAG_NONE;

        Microsoft::WRL::ComPtr<ID3D12Resource> res;
        if (FAILED(m_device.device()->CreateCommittedResource(
            &hp, D3D12_HEAP_FLAG_SHARED, &rd,
            D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&res)))) {
            return false;
        }

        int srvIndex = m_device.createBufferSRV(res.Get(), numElements, strideBytes);
        if (srvIndex < 0) return false;

        HANDLE handle = nullptr;
        if (FAILED(m_device.device()->CreateSharedHandle(res.Get(), nullptr, GENERIC_ALL, nullptr, &handle))) {
            return false;
        }

        m_sharedParticleBuffer = res;
        m_particleSrvIndex = srvIndex;
        m_particleCount = numElements;
        outSharedHandle = handle;
        return true;
    }

    void RendererD3D12::BuildFrameGraph() {
        m_fg = core::FrameGraph{};

        auto rsGfx = CreateRootSignatureGfx(m_device.device());

        // 关键：传入仓库内的相对路径，由 Compile 自行解析到绝对路径
        auto vs = Compile(L"engine\\gfx\\d3d12_shaders\\points.hlsl", "VSMain", "vs_5_1");
        auto ps = Compile(L"engine\\gfx\\d3d12_shaders\\points.hlsl", "PSMain", "ps_5_1");

        Microsoft::WRL::ComPtr<ID3D12PipelineState> psoPoints;
        if (rsGfx && vs && ps) {
            DXGI_FORMAT rtFmt = m_device.currentBackbuffer()->GetDesc().Format;
            D3D12_GRAPHICS_PIPELINE_STATE_DESC pso{};
            pso.pRootSignature = rsGfx.Get();
            pso.VS = { vs->GetBufferPointer(), vs->GetBufferSize() };
            pso.PS = { ps->GetBufferPointer(), ps->GetBufferSize() };
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
            if (FAILED(m_device.device()->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(&psoPoints)))) {
                psoPoints.Reset();
                OutputDebugStringA("[PSO] CreateGraphicsPipelineState failed for points pipeline.\n");
            }
        }
        else {
            OutputDebugStringA("[PSO] Points shaders or root signature missing, points pass will be skipped.\n");
        }

        core::PassDesc clear{}; clear.name = "clear";
        clear.execute = [this]() {
            m_device.beginFrame();
            std::memcpy(m_clearColor, m_visual.clearColor, sizeof(m_clearColor));
            m_device.clearCurrentRTV(m_clearColor);
            m_device.writeTimestamp();
            };
        m_fg.addPass(clear);

        if (rsGfx && psoPoints) {
            core::PassDesc points{}; points.name = "points";
            points.execute = [this, rsGfx, psoPoints]() {
                auto* cl = m_device.cmdList();

                auto rtvHandle = m_device.currentRTV();
                cl->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);

                D3D12_VIEWPORT vp{ 0.0f, 0.0f, float(m_device.width()), float(m_device.height()), 0.0f, 1.0f };
                D3D12_RECT sc{ 0,0,(LONG)m_device.width(), (LONG)m_device.height() };
                cl->RSSetViewports(1, &vp);
                cl->RSSetScissorRects(1, &sc);

                ID3D12DescriptorHeap* heaps[] = { m_device.srvHeap() };
                cl->SetDescriptorHeaps(1, heaps);
                cl->SetGraphicsRootSignature(rsGfx.Get());
                cl->SetPipelineState(psoPoints.Get());
                cl->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

                if (m_sharedParticleBuffer) {
                    D3D12_RESOURCE_BARRIER b{};
                    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                    b.Transition.pResource = m_sharedParticleBuffer.Get();
                    b.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
                    b.Transition.StateAfter = D3D12_RESOURCE_STATE_GENERIC_READ;
                    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                    cl->ResourceBarrier(1, &b);
                }

                PerFrameCB cb{};
                float3 eye = m_camera.eye;
                float3 at = m_camera.at;
                float3 up = m_camera.up;
                Mat4 V = lookAtRH(eye, at, up);
                float aspect = (m_device.height() > 0) ? (float)m_device.width() / (float)m_device.height() : 1.0f;
                Mat4 P = perspectiveFovRH_ZO(Deg2Rad(m_camera.fovYDeg), aspect, m_camera.nearZ, m_camera.farZ);
                Mat4 VP = mul(P, V);

                std::memcpy(cb.viewProj, VP.m, sizeof(cb.viewProj));
                cb.screenSize[0] = (float)m_device.width();
                cb.screenSize[1] = (float)m_device.height();
                cb.particleRadiusPx = m_visual.particleRadiusPx;
                cb.thicknessScale = m_visual.thicknessScale;
                cl->SetGraphicsRoot32BitConstants(0, sizeof(PerFrameCB) / 4, &cb, 0);

                if (m_particleSrvIndex >= 0 && m_particleCount > 0) {
                    auto gpuH = m_device.srvGpuHandleAt((uint32_t)m_particleSrvIndex);
                    cl->SetGraphicsRootDescriptorTable(1, gpuH);
                    UINT instanceCount = m_particleCount;
                    cl->DrawInstanced(6, instanceCount, 0, 0);
                }

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

        core::PassDesc present{}; present.name = "present";
        present.execute = [this]() { m_device.present(); };
        m_fg.addPass(present);

        m_fg.compile();
    }

    void RendererD3D12::RenderFrame(core::Profiler& profiler) {
        std::vector<double> gpuMs;
        m_fg.execute([&](const std::string& name, double ms) { profiler.addRow(name, ms); });
        if (m_device.readbackPassTimesMs(gpuMs)) {
            for (size_t i = 0; i < gpuMs.size(); ++i) profiler.addRow(std::string("gpu_") + std::to_string(i), gpuMs[i]);
        }
    }

    bool RendererD3D12::ImportSharedBufferAsSRV(HANDLE sharedHandle, uint32_t numElements, uint32_t strideBytes, int& outSrvIndex) {
        ID3D12Resource* res = nullptr;
        if (!m_device.openSharedResource(sharedHandle, __uuidof(ID3D12Resource), reinterpret_cast<void**>(&res))) return false;
        m_sharedParticleBuffer.Attach(res);
        m_particleSrvIndex = m_device.createBufferSRV(m_sharedParticleBuffer.Get(), numElements, strideBytes);
        if (m_particleSrvIndex < 0) return false;
        m_particleCount = numElements;
        outSrvIndex = m_particleSrvIndex;
        return true;
    }

} // namespace gfx