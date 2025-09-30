#include "renderer.h"
#include <chrono>
#include <d3dcompiler.h>
#include <cstring>

#pragma comment(lib, "d3dcompiler.lib")

namespace gfx {

    namespace {

        // Root signature: b0 as root constants, table0: SRV t0 (ParticlePos)
        static Microsoft::WRL::ComPtr<ID3D12RootSignature> CreateRootSignatureGfx(ID3D12Device* dev) {
            using Microsoft::WRL::ComPtr;

            D3D12_DESCRIPTOR_RANGE1 ranges[1] = {};
            ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            ranges[0].NumDescriptors = 1; // ParticlePos SRV
            ranges[0].BaseShaderRegister = 0; // t0
            ranges[0].RegisterSpace = 0;
            ranges[0].Flags = D3D12_DESCRIPTOR_RANGE_FLAG_NONE;
            ranges[0].OffsetInDescriptorsFromTableStart = 0;

            D3D12_ROOT_PARAMETER1 params[2] = {};
            params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; // PerFrameCB
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
            ComPtr<ID3D12RootSignature> rs;
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

    } // namespace

    bool RendererD3D12::Initialize(HWND hwnd, const RenderInitParams& p) {
        DeviceInitParams dp; dp.width = p.width; dp.height = p.height; dp.bufferCount = 3;
        if (!m_device.initialize(hwnd, dp)) return false;
        // Create SRV heap for interop SRVs
        m_device.createSrvHeap(256, true);
        BuildFrameGraph();
        return true;
    }

    void RendererD3D12::Shutdown() {
        m_sharedParticleBuffer.Reset();
        m_device.shutdown();
    }

    static Microsoft::WRL::ComPtr<ID3DBlob> Compile(const wchar_t* path, const char* entry, const char* target) {
        UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#if defined(_DEBUG)
        flags |= D3DCOMPILE_DEBUG;
#endif
        Microsoft::WRL::ComPtr<ID3DBlob> cs, err;
        if (FAILED(D3DCompileFromFile(path, nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, entry, target, flags, 0, &cs, &err))) {
            if (err) OutputDebugStringA((char*)err->GetBufferPointer());
            return nullptr;
        }
        return cs;
    }

    void RendererD3D12::BuildFrameGraph() {
        m_fg = core::FrameGraph{};

        // Root signature for points
        auto rsGfx = CreateRootSignatureGfx(m_device.device());

        // Compile points shaders
        auto vs = Compile(L"engine/gfx/d3d12_shaders/points.hlsl", "VSMain", "vs_5_1");
        auto ps = Compile(L"engine/gfx/d3d12_shaders/points.hlsl", "PSMain", "ps_5_1");

        // PSO for points (render directly to backbuffer)
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
            }
        }

        // Clear pass (backbuffer)
        core::PassDesc clear{}; clear.name = "clear";
        clear.execute = [this]() {
            m_device.beginFrame();
            m_device.clearCurrentRTV(m_clearColor);
            m_device.writeTimestamp();
            };
        m_fg.addPass(clear);

        // Points pass
        if (rsGfx && psoPoints) {
            core::PassDesc points{}; points.name = "points";
            points.execute = [this, rsGfx, psoPoints]() {
                auto* cl = m_device.cmdList();

                // Set RTV to current backbuffer (already in RT state by clear)
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

                // Per-frame constants
                PerFrameCB cb{};
                for (int i = 0; i < 16; ++i) cb.viewProj[i] = (i % 5) == 0 ? 1.0f : 0.0f; // TODO: set real ViewProj
                cb.screenSize[0] = (float)m_device.width();
                cb.screenSize[1] = (float)m_device.height();
                cb.particleRadiusPx = 3.0f; // TODO: parameterize via config
                cb.thicknessScale = 1.0f;
                cl->SetGraphicsRoot32BitConstants(0, sizeof(PerFrameCB) / 4, &cb, 0);

                if (m_particleSrvIndex >= 0 && m_particleCount > 0) {
                    auto gpuH = m_device.srvGpuHandleAt((uint32_t)m_particleSrvIndex);
                    cl->SetGraphicsRootDescriptorTable(1, gpuH);
                    UINT instanceCount = m_particleCount;
                    cl->DrawInstanced(6, instanceCount, 0, 0);
                }
                m_device.writeTimestamp();
                };
            m_fg.addPass(points);
        }

        // Present
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
