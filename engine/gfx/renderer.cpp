#include "renderer.h"
#include "../core/prof_nvtx.h"
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
        static Mat4 mul(const Mat4& a, const Mat4& b) { Mat4 r{}; for(int c=0;c<4;++c) for(int r0=0;r0<4;++r0) r.m[c*4+r0]=
            a.m[0*4+r0]*b.m[c*4+0]+a.m[1*4+r0]*b.m[c*4+1]+a.m[2*4+r0]*b.m[c*4+2]+a.m[3*4+r0]*b.m[c*4+3]; return r; }
        static Mat4 lookAtRH(float3 eye,float3 at,float3 up){ float3 z=make_float3(eye.x-at.x,eye.y-at.y,eye.z-at.z);
            float lenz=sqrtf(z.x*z.x+z.y*z.y+z.z*z.z); z.x/=lenz; z.y/=lenz; z.z/=lenz;
            float3 x=make_float3(up.y*z.z-up.z*z.y, up.z*z.x-up.x*z.z, up.x*z.y-up.y*z.x);
            float lenx=sqrtf(x.x*x.x+x.y*x.y+x.z*x.z); x.x/=lenx; x.y/=lenx; x.z/=lenx;
            float3 y=make_float3(z.y*x.z - z.z*x.y, z.z*x.x - z.x*x.z, z.x*x.y - z.y*x.x);
            Mat4 m{}; m.m[0]=x.x; m.m[4]=x.y; m.m[8]=x.z;  m.m[12]=-(x.x*eye.x+x.y*eye.y+x.z*eye.z);
            m.m[1]=y.x; m.m[5]=y.y; m.m[9]=y.z;  m.m[13]=-(y.x*eye.x+y.y*eye.y+y.z*eye.z);
            m.m[2]=z.x; m.m[6]=z.y; m.m[10]=z.z; m.m[14]=-(z.x*eye.x+z.y*eye.y+z.z*eye.z);
            m.m[3]=0.f; m.m[7]=0.f; m.m[11]=0.f; m.m[15]=1.f; return m; }
        static Mat4 perspectiveFovRH_ZO(float fovy,float aspect,float zn,float zf){
            float f=1.0f/tanf(fovy*0.5f); Mat4 m{}; m.m[0]=f/aspect; m.m[5]=f; m.m[10]=zf/(zn-zf); m.m[14]=(zf*zn)/(zn-zf); m.m[11]=-1.0f; return m; }
        static float Deg2Rad(float deg){ return deg*3.14159265358979323846f/180.0f; }

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

        // 修复版 RootSignature：常量 + 粘贴 SRV(t0) + 调色板 SRV(t1)
        static Microsoft::WRL::ComPtr<ID3D12RootSignature> CreateRootSignatureGfx(ID3D12Device* dev) {
            using Microsoft::WRL::ComPtr;

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

            D3D12_ROOT_PARAMETER1 params[3] = {};
            // b0 constants
            params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            params[0].Constants.ShaderRegister = 0;
            params[0].Constants.RegisterSpace = 0;
            params[0].Constants.Num32BitValues = sizeof(PerFrameCB)/4;
            // t0 particles
            params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            D3D12_ROOT_DESCRIPTOR_TABLE1 tbl0{};
            tbl0.NumDescriptorRanges = 1;
            tbl0.pDescriptorRanges = &rangeParticles;
            params[1].DescriptorTable = tbl0;
            // t1 palette
            params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            D3D12_ROOT_DESCRIPTOR_TABLE1 tbl1{};
            tbl1.NumDescriptorRanges = 1;
            tbl1.pDescriptorRanges = &rangePalette;
            params[2].DescriptorTable = tbl1;

            D3D12_VERSIONED_ROOT_SIGNATURE_DESC vdesc{};
            vdesc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
            vdesc.Desc_1_1.NumParameters = _countof(params);
            vdesc.Desc_1_1.pParameters = params;
            vdesc.Desc_1_1.NumStaticSamplers = 0;
            vdesc.Desc_1_1.pStaticSamplers = nullptr;
            vdesc.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

            ComPtr<ID3DBlob> blob, err;
            if (FAILED(D3D12SerializeVersionedRootSignature(&vdesc, &blob, &err))) {
                if (err) OutputDebugStringA((char*)err->GetBufferPointer());
                return nullptr;
            }
            ComPtr<ID3D12RootSignature> rs;
            if (FAILED(dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&rs)))) {
                return nullptr;
            }
            return rs;
        }

        D3D12_BLEND_DESC AlphaBlendDesc(){ D3D12_BLEND_DESC b{}; b.AlphaToCoverageEnable=FALSE; b.IndependentBlendEnable=FALSE;
            auto& rt=b.RenderTarget[0]; rt.BlendEnable=TRUE; rt.SrcBlend=D3D12_BLEND_SRC_ALPHA; rt.DestBlend=D3D12_BLEND_INV_SRC_ALPHA;
            rt.BlendOp=D3D12_BLEND_OP_ADD; rt.SrcBlendAlpha=D3D12_BLEND_ONE; rt.DestBlendAlpha=D3D12_BLEND_INV_SRC_ALPHA;
            rt.BlendOpAlpha=D3D12_BLEND_OP_ADD; rt.RenderTargetWriteMask=D3D12_COLOR_WRITE_ENABLE_ALL; return b; }
        D3D12_RASTERIZER_DESC DefaultRasterizerNoCull(){ D3D12_RASTERIZER_DESC r{}; r.FillMode=D3D12_FILL_MODE_SOLID; r.CullMode=D3D12_CULL_MODE_NONE;
            r.FrontCounterClockwise=FALSE; r.DepthBias=D3D12_DEFAULT_DEPTH_BIAS; r.DepthBiasClamp=D3D12_DEFAULT_DEPTH_BIAS_CLAMP;
            r.SlopeScaledDepthBias=D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS; r.DepthClipEnable=TRUE; r.MultisampleEnable=FALSE;
            r.AntialiasedLineEnable=FALSE; r.ForcedSampleCount=0; r.ConservativeRaster=D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF; return r; }

        // 编译前先 ResolveShaderPath
        static Microsoft::WRL::ComPtr<ID3DBlob> Compile(const wchar_t* relativePath, const char* entry, const char* target) {
            UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#if defined(_DEBUG)
            flags|=D3DCOMPILE_DEBUG;
#endif
            std::wstring full = ResolveShaderPath(relativePath);
            // 手工读取文件，过滤 UTF-8 BOM / 非法首字节
            std::ifstream ifs(full, std::ios::binary);
            if (!ifs.is_open()) {
                OutputDebugStringW((L"[HLSL] Open failed: " + full + L"\n").c_str());
                return nullptr;
            }
            std::vector<char> bytes((std::istreambuf_iterator<char>(ifs)),
            std::istreambuf_iterator<char>());
            if (bytes.empty()) {
                OutputDebugStringW((L"[HLSL] Empty file: " + full + L"\n").c_str());
                return nullptr;
            }
            // 移除 UTF-8 BOM
                if (bytes.size() >= 3 &&
                    (unsigned char)bytes[0] == 0xEF &&
                    (unsigned char)bytes[1] == 0xBB &&
                    (unsigned char)bytes[2] == 0xBF) {
                bytes.erase(bytes.begin(), bytes.begin() + 3);
            }
            // 过滤首个不可打印控制字符
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
            HRESULT hr = D3DCompile(bytes.data(), bytes.size(),
                /*sourceName*/nullptr,
                /*defines*/nullptr,
                D3D_COMPILE_STANDARD_FILE_INCLUDE,
                entry, target, flags, 0,
                &cs, &err);
            if (FAILED(hr)) {
                if (err) OutputDebugStringA((char*)err->GetBufferPointer());
                std::wstring msg = L"[HLSL] Compile failed: " + full + L"\n";
                OutputDebugStringW(msg.c_str());
                // 输出前几个字节的十六进制辅助定位
                char hexBuf[256]; int pos = 0;
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

    bool RendererD3D12::Initialize(HWND hwnd,const RenderInitParams& p){
        prof::Range r("Renderer.Initialize", prof::Color(0x20, 0x80, 0xC0));
        DeviceInitParams dp; dp.width=p.width; dp.height=p.height; dp.bufferCount=3; dp.vsync=p.vsync;
        if(!m_device.initialize(hwnd,dp)) return false;
        m_device.createSrvHeap(256,true);
        std::memcpy(m_clearColor,m_visual.clearColor,sizeof(m_clearColor));
        // 新增：创建共享时间线 fence 供 CUDA 导入
        if (FAILED(m_device.device()->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&m_timelineFence)))) return false;
        if (FAILED(m_device.device()->CreateSharedHandle(m_timelineFence.Get(), nullptr, GENERIC_ALL, nullptr, &m_timelineFenceSharedHandle))) return false;
        m_renderFenceValue = 0; // 渲染完成值（偶数序列）
        m_lastSimFenceValue = 0; // 最近模拟完成值（奇数）
        BuildFrameGraph();
        return true;
    }

    void RendererD3D12::Shutdown(){
        m_paletteBuffer.Reset();
        if (m_timelineFenceSharedHandle) { CloseHandle(m_timelineFenceSharedHandle); m_timelineFenceSharedHandle=nullptr; }
        m_timelineFence.Reset();
        m_device.shutdown();
    }

    // 新增：等待指定模拟完成 fence 值（只排队 GPU 等待）
    void RendererD3D12::WaitSimulationFence(uint64_t simValue){
        if(!m_timelineFence) return;
        if (simValue > m_lastSimFenceValue) {
            m_device.queue()->Wait(m_timelineFence.Get(), simValue);
            m_lastSimFenceValue = simValue; // 记录最新已等待的模拟值
        }
    }

    // 新增：渲染完成后递增渲染 fence（偶数），供模拟可选使用
    void RendererD3D12::SignalRenderComplete(uint64_t /*lastSimValue*/){
        if(!m_timelineFence) return;
        ++m_renderFenceValue; // 偶数序列：0,1,2,... 这里不强制偶数，仅单调即可
        m_device.queue()->Signal(m_timelineFence.Get(), m_renderFenceValue);
    }

    bool RendererD3D12::CreateSharedParticleBufferIndexed(int slot, uint32_t numElements, uint32_t strideBytes, HANDLE& outSharedHandle) {
        outSharedHandle = nullptr;
        if (slot < 0 || slot > 1) return false;
        const UINT64 sizeBytes = UINT64(numElements) * UINT64(strideBytes);
        D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = sizeBytes;
        rd.Height = 1;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.Format = DXGI_FORMAT_UNKNOWN;
        rd.SampleDesc = { 1,0 };
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        rd.Flags = D3D12_RESOURCE_FLAG_NONE;

        Microsoft::WRL::ComPtr<ID3D12Resource> res;
        if (FAILED(m_device.device()->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &rd,
            D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&res)))) return false;

        int srvIndex = m_device.createBufferSRV(res.Get(), numElements, strideBytes);
        if (srvIndex < 0) return false;

        HANDLE handle = nullptr;
        if (FAILED(m_device.device()->CreateSharedHandle(res.Get(), nullptr, GENERIC_ALL, nullptr, &handle))) return false;

        m_sharedParticleBuffers[slot] = res;
        m_particleSrvIndexPing[slot] = srvIndex;
        if (slot == 0) m_activePingIndex = 0;

        // 不在此更新 m_particleSrvIndex（由 UpdateParticleSRVForPingPong 决定）
        if (numElements > m_particleCount) m_particleCount = numElements;
        outSharedHandle = handle;
        return true;
    }

    void RendererD3D12::RegisterPingPongCudaPtrs(const void* ptrA, const void* ptrB) {
        m_knownCudaPtrs[0] = const_cast<void*>(ptrA);
        m_knownCudaPtrs[1] = const_cast<void*>(ptrB);
        if (ptrA) {
            UpdateParticleSRVForPingPong(ptrA);
        }
        std::fprintf(stderr,
            "[Render.PingPong][Register] ptrA=%p ptrB=%p activePing=%d srvIndex=%d srvA=%d srvB=%d\n",
            ptrA, ptrB, m_activePingIndex, m_particleSrvIndexPing[m_activePingIndex],
            m_particleSrvIndexPing[0], m_particleSrvIndexPing[1]);
    }

    void RendererD3D12::UpdateParticleSRVForPingPong(const void* devicePtrCurr) {
        if (!devicePtrCurr) return;
        // 匹配保存的 CUDA 指针 → 选择 SRV
        for (int i = 0; i < 2; ++i) {
            if (m_knownCudaPtrs[i] == devicePtrCurr && m_particleSrvIndexPing[i] >= 0) {
                m_activePingIndex = i;
                return;
            }
        }
        // 若未匹配到，保持原值（可能仍是单缓冲模式）
    }

    void RendererD3D12::BuildFrameGraph(){
        prof::Range r("Renderer.BuildFrameGraph", prof::Color(0x90,0x40,0x30));
        m_fg = core::FrameGraph{};
        auto rsGfx = CreateRootSignatureGfx(m_device.device());
        // 编译 float版着色器
        auto vsFloat = Compile(L"engine\\gfx\\d3d12_shaders\\points.hlsl","VSMain","vs_5_1");
        auto psFloat = Compile(L"engine\\gfx\\d3d12_shaders\\points.hlsl","PSMain","ps_5_1");
        // 编译 half版（独立文件 points_half.hlsl，用 uint2 -> four fp16 解码）
        auto vsHalf = Compile(L"engine\\gfx\\d3d12_shaders\\points_half.hlsl","VSMain","vs_5_1");
        auto psHalf = Compile(L"engine\\gfx\\d3d12_shaders\\points_half.hlsl","PSMain","ps_5_1");

        m_psoPointsFloat.Reset();
        m_psoPointsHalf.Reset();
        if(rsGfx && vsFloat && psFloat){
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
            pso.InputLayout = { nullptr,0 };
            pso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            pso.NumRenderTargets =1;
            pso.RTVFormats[0] = rtFmt;
            pso.SampleDesc.Count =1;
            if (SUCCEEDED(m_device.device()->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(&m_psoPointsFloat))))
                OutputDebugStringA("[PSO] Created float points pipeline.\n");
        }
        if(rsGfx && vsHalf && psHalf){
            DXGI_FORMAT rtFmt = m_device.currentBackbuffer()->GetDesc().Format;
            D3D12_GRAPHICS_PIPELINE_STATE_DESC pso{};
            pso.pRootSignature = rsGfx.Get();
            pso.VS = { vsHalf->GetBufferPointer(), vsHalf->GetBufferSize() };
            pso.PS = { psHalf->GetBufferPointer(), psHalf->GetBufferSize() };
            pso.BlendState = AlphaBlendDesc();
            pso.SampleMask = UINT_MAX;
            pso.RasterizerState = DefaultRasterizerNoCull();
            pso.DepthStencilState.DepthEnable = FALSE;
            pso.DepthStencilState.StencilEnable = FALSE;
            pso.InputLayout = { nullptr,0 };
            pso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            pso.NumRenderTargets =1;
            pso.RTVFormats[0] = rtFmt;
            pso.SampleDesc.Count =1;
            if (SUCCEEDED(m_device.device()->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(&m_psoPointsHalf))))
                OutputDebugStringA("[PSO] Created half points pipeline.\n");
        }

        core::PassDesc clear{}; clear.name="clear";
        clear.execute = [this](){
            m_device.beginFrame();
            std::memcpy(m_clearColor,m_visual.clearColor,sizeof(m_clearColor));
            m_device.clearCurrentRTV(m_clearColor);
            m_device.writeTimestamp();
        };
        m_fg.addPass(clear);

        // 替换旧 points pass，选择 PSO
        if(rsGfx && (m_psoPointsFloat || m_psoPointsHalf)){
            core::PassDesc points{}; points.name = "points";
            points.execute = [this, rsGfx]() {
                auto* cl = m_device.cmdList();
                auto rtvHandle = m_device.currentRTV();
                cl->OMSetRenderTargets(1,&rtvHandle,FALSE,nullptr);
                D3D12_VIEWPORT vp{0.f,0.f,(float)m_device.width(),(float)m_device.height(),0.f,1.f};
                D3D12_RECT sc{0,0,(LONG)m_device.width(),(LONG)m_device.height()};
                cl->RSSetViewports(1,&vp); cl->RSSetScissorRects(1,&sc);
                ID3D12DescriptorHeap* heaps[] = { m_device.srvHeap() }; cl->SetDescriptorHeaps(1, heaps);
                cl->SetGraphicsRootSignature(rsGfx.Get());
                bool useHalf = m_useHalfRender && m_psoPointsHalf; //运行时标志
                cl->SetPipelineState(useHalf ? m_psoPointsHalf.Get() : m_psoPointsFloat.Get());
                cl->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                PerFrameCB cb{}; Mat4 V=lookAtRH(m_camera.eye,m_camera.at,m_camera.up); float aspect=(m_device.height()>0)?(float)m_device.width()/(float)m_device.height():1.0f;
                Mat4 P=perspectiveFovRH_ZO(Deg2Rad(m_camera.fovYDeg),aspect,m_camera.nearZ,m_camera.farZ); Mat4 VP=mul(P,V); std::memcpy(cb.viewProj,VP.m,sizeof(cb.viewProj));
                cb.screenSize[0]=(float)m_device.width(); cb.screenSize[1]=(float)m_device.height();
                cb.particleRadiusPx=m_visual.particleRadiusPx; cb.thicknessScale=m_visual.thicknessScale;
                cb.groups=m_groups; cb.particlesPerGroup=m_particlesPerGroup;
                cl->SetGraphicsRoot32BitConstants(0,sizeof(PerFrameCB)/4,&cb,0);
                int currentSrv = (m_activePingIndex>=0)? m_particleSrvIndexPing[m_activePingIndex] : -1;
                if(currentSrv>=0 && m_particleCount>0){ auto gpuPos=m_device.srvGpuHandleAt((uint32_t)currentSrv); cl->SetGraphicsRootDescriptorTable(1,gpuPos); }
                if(m_paletteSrvIndex>=0){ auto gpuPal=m_device.srvGpuHandleAt((uint32_t)m_paletteSrvIndex); cl->SetGraphicsRootDescriptorTable(2,gpuPal); }
                UINT instanceCount=m_particleCount; if(instanceCount>0) cl->DrawInstanced(6,instanceCount,0,0);
                m_device.writeTimestamp();
            };
            m_fg.addPass(points);
        }

        core::PassDesc present{}; present.name="present";
        present.execute = [this](){ m_device.present(); };
        m_fg.addPass(present);

        m_fg.compile();
    }

    void RendererD3D12::RenderFrame(core::Profiler& profiler) {
        prof::Range r("Renderer.RenderFrame", prof::Color(0x40, 0x90, 0x50));
        // 保持 m_useHalfRender（由资源导入或外部接口控制）
        std::vector<double> gpuMs;
        m_fg.execute([&](const std::string& name, double ms) { profiler.addRow(name, ms); });
        if (m_device.readbackPassTimesMs(gpuMs))
            for (size_t i = 0; i < gpuMs.size(); ++i) profiler.addRow(std::string("gpu_") + std::to_string(i), gpuMs[i]);
    }

    void RendererD3D12::WaitForGPU(){ 
        prof::Range r("Renderer.WaitForGPU", prof::Color(0xAA, 0x20, 0x20));
        m_device.waitForGPU();
    }

    bool RendererD3D12::UpdateGroupPalette(const float* rgbTriples,uint32_t groupCount){
        if(!rgbTriples || groupCount==0){
            m_groups=0; m_paletteSrvIndex=-1; m_paletteBuffer.Reset(); return true;
        }
        size_t bytes = sizeof(float) * 3 * groupCount;
        // 直接使用上传堆，避免初始化阶段显式提交命令的问题
        D3D12_HEAP_PROPERTIES hpUp{}; hpUp.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC rd{}; rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = bytes; rd.Height = 1; rd.DepthOrArraySize = 1;
        rd.MipLevels = 1; rd.Format = DXGI_FORMAT_UNKNOWN;
        rd.SampleDesc = { 1,0 }; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        rd.Flags = D3D12_RESOURCE_FLAG_NONE;
        
        Microsoft::WRL::ComPtr<ID3D12Resource> buf;
        if (FAILED(m_device.device()->CreateCommittedResource(&hpUp, D3D12_HEAP_FLAG_NONE, &rd,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&buf)))) return false;

        void* mapped = nullptr; D3D12_RANGE range{ 0,0 };
        if (FAILED(buf->Map(0, &range, &mapped))) return false;
        std::memcpy(mapped, rgbTriples, bytes);
        buf->Unmap(0, nullptr);

        m_paletteBuffer = buf;
        m_paletteSrvIndex = m_device.createBufferSRV(m_paletteBuffer.Get(), groupCount, sizeof(float) * 3);
        return (m_paletteSrvIndex >= 0);
        return (m_paletteSrvIndex >= 0);
    }

} // namespace gfx