#include "simulator.h"
#include "math_utils.h"
#include "poisson_disk.h"
#include "numeric_utils.h"
#include "emit_params.h"
#include "stats.h"
#include "logging.h"
#include "emitter.h"
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <random>
#include <cuda_fp16.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <unordered_set>
#include "grid_system.h"
#include "../engine/core/console.h"
#include "../engine/core/prof_nvtx.h"
#include "precision_traits.cuh"
#include "device_globals.cuh"
#include "precision_stage.h"
#include "simulation_context.h"
#include <limits>
#include "../engine/gfx/renderer.h"
#include "grid_strategy_dense.h"
#include "device_pos_state.cuh"
#include "grid_strategy_hashed.h"
#include "graph_builder.h"
#include "phase_pipeline.h"
#include "post_ops.h"

#ifndef CUDA_CHECK
#define CUDA_CHECK(expr) \
    do { \
        cudaError_t _err = (expr); \
        if (_err != cudaSuccess) { \
            sim::Log(sim::LogChannel::Error, "CUDA %s (%d)", cudaGetErrorString(_err), (int)_err); \
        } \
    } while (0)
#endif

// ===================== D2D Memcpy 调试插桩 =====================
// 说明：所有设备到设备整块复制统一通过下列宏记录来源。
// 输出条件：console.debug.printDiagnostics 或 console.debug.printWarnings 为 true
// 搜索关键字: [D2D-Memcpy]
namespace sim {
    extern uint64_t g_simFrameIndex;
    static inline bool ShouldLogD2D() {
        const auto& cc = console::Instance();
        return (cc.debug.printDiagnostics || cc.debug.printWarnings);
    }
    static inline void LogD2DMemcpy(const char* tag,
                                    const void* dst,
                                    const void* src,
                                    size_t bytes,
                                    bool async) {
        if (!ShouldLogD2D()) return;
        std::fprintf(stderr,
            "[D2D-Memcpy] frame=%llu tag=%s bytes=%zu dst=%p src=%p async=%d\n",
            (unsigned long long)g_simFrameIndex, tag, bytes, dst, src, async ? 1 : 0);
    }
#ifdef _WIN32
    static inline bool hostPtrReadable(const void* p) {
        if (!p) return false;
        MEMORY_BASIC_INFORMATION mbi{};
        if (VirtualQuery(p, &mbi, sizeof(mbi)) == 0) return false;
        if (mbi.State != MEM_COMMIT) return false;
        if (mbi.Protect & PAGE_NOACCESS) return false;
        if (mbi.Protect & PAGE_GUARD) return false;
        return true;
    }
#else
    static inline bool hostPtrReadable(const void* p) { return p != nullptr; }
#endif
} // namespace sim

#define CUDA_LOGGED_MEMCPY_D2D(TAG, DST, SRC, BYTES) \
    do { \
        sim::LogD2DMemcpy(TAG, (const void*)(DST), (const void*)(SRC), (size_t)(BYTES), false); \
        CUDA_CHECK(cudaMemcpy((DST), (SRC), (BYTES), cudaMemcpyDeviceToDevice)); \
    } while (0)

#define CUDA_LOGGED_MEMCPY_D2D_ASYNC(TAG, DST, SRC, BYTES, STREAM) \
    do { \
        sim::LogD2DMemcpy(TAG, (const void*)(DST), (const void*)(SRC), (size_t)(BYTES), true); \
        CUDA_CHECK(cudaMemcpyAsync((DST), (SRC), (BYTES), cudaMemcpyDeviceToDevice, (STREAM))); \
    } while (0)
// ===================== End D2D Memcpy 插桩 =====================

// ===== 外部 CUDA kernel =====
extern "C" void LaunchHashKeys(uint32_t*, uint32_t*, const float4*, sim::GridBounds, uint32_t, cudaStream_t);
extern "C" void LaunchHashKeysMP(uint32_t*, uint32_t*, const float4*, const sim::Half4*, sim::GridBounds, uint32_t, cudaStream_t);
extern "C" void LaunchCellRanges(uint32_t*, uint32_t*, const uint32_t*, uint32_t, uint32_t, cudaStream_t);
extern "C" void LaunchCellRangesCompact(uint32_t*, uint32_t*, uint32_t*, const uint32_t*, uint32_t, cudaStream_t);
extern "C" void LaunchXSPH(float4*, const float4*, const float4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchXSPHCompact(float4*, const float4*, const float4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchXSPHCompactMP(float4*, const float4*, const sim::Half4*, const float4*, const sim::Half4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, uint32_t, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchXSPHMP(float4*, const float4*, const sim::Half4*, const float4*, const sim::Half4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" bool EnsureCellCompactScratch(uint32_t, uint32_t);
extern "C" void LaunchSortPairsQuery(size_t*, const uint32_t*, uint32_t*, const uint32_t*, uint32_t*, uint32_t, cudaStream_t);
extern "C" void LaunchSortPairs(void*, size_t, uint32_t*, uint32_t*, uint32_t*, uint32_t*, uint32_t, cudaStream_t);
extern "C" void LaunchRecycleToNozzleConst(float4*, float4*, float4*, sim::GridBounds, float, uint32_t, int, cudaStream_t);
extern "C" void LaunchIntegratePredGlobals(float3 gravity, float dt, uint32_t N, cudaStream_t);
extern "C" void LaunchVelocityGlobals(float dtInv, uint32_t N, cudaStream_t);
extern "C" void LaunchBoundaryGlobals(sim::GridBounds, float restitution, uint32_t N, bool xsphApplied, cudaStream_t);
extern "C" void LaunchLambdaDenseGlobals(const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchLambdaCompactGlobals(const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchDeltaApplyDenseGlobals(const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchDeltaApplyCompactGlobals(const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchXsphDenseGlobals(const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchXsphCompactGlobals(const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);

extern "C" void* GetRecycleKernelPtr();

namespace sim {
    uint64_t g_simFrameIndex = 0;

    // 新增：渲染半精外部缓冲导入
    bool Simulator::importRenderHalfBuffer(void* sharedHandleWin32, size_t bytes) {
        if (!sharedHandleWin32 || bytes == 0) return false;
        if (m_extRenderHalf) {
            cudaDestroyExternalMemory(m_extRenderHalf);
            m_extRenderHalf = nullptr;
            m_renderHalfMappedPtr = nullptr;
            m_renderHalfBytes = 0;
        }
        cudaExternalMemoryHandleDesc memDesc{};
        memDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
        memDesc.handle.win32.handle = sharedHandleWin32;
        memDesc.size = bytes;
        memDesc.flags = cudaExternalMemoryDedicated;
        if (cudaImportExternalMemory(&m_extRenderHalf, &memDesc) != cudaSuccess) {
            std::fprintf(stderr, "[RenderHalf][Import] failed handle=%p bytes=%zu\n", sharedHandleWin32, bytes);
            return false;
        }
        cudaExternalMemoryBufferDesc bufDesc{}; bufDesc.offset = 0; bufDesc.size = bytes;
        void* devPtr = nullptr;
        if (cudaExternalMemoryGetMappedBuffer(&devPtr, m_extRenderHalf, &bufDesc) != cudaSuccess) {
            std::fprintf(stderr, "[RenderHalf][Map] failed\n");
            cudaDestroyExternalMemory(m_extRenderHalf); m_extRenderHalf = nullptr; return false;
        }
        m_renderHalfMappedPtr = devPtr;
        m_renderHalfBytes = bytes;
        std::fprintf(stderr, "[RenderHalf][Ready] mappedPtr=%p bytes=%zu\n", m_renderHalfMappedPtr, m_renderHalfBytes);
        return true;
    }

    void Simulator::publishRenderHalf(uint32_t count) {
        if (!m_renderHalfMappedPtr || !m_extRenderHalf || count == 0) return;
        if (!m_bufs.useRenderHalf || !m_bufs.d_render_pos_h4) return;
        size_t needBytes = size_t(count) * sizeof(sim::Half4);
        if (needBytes > m_renderHalfBytes) {
            std::fprintf(stderr, "[RenderHalf][Warn] external buffer too small need=%zu have=%zu\n", needBytes, m_renderHalfBytes); return;
        }
        // 始终 pack 当前 curr（若 position 已 half 可直接复制）
        if (m_bufs.usePosHalf && m_bufs.d_pos_curr_h4) {
            CUDA_LOGGED_MEMCPY_D2D_ASYNC("RenderHalf.Publish.Direct", m_renderHalfMappedPtr, m_bufs.d_pos_curr_h4, needBytes, m_stream);
        }
        else {
            m_bufs.packRenderToHalf(count, m_stream);
            CUDA_LOGGED_MEMCPY_D2D_ASYNC("RenderHalf.Publish.Pack", m_renderHalfMappedPtr, m_bufs.d_render_pos_h4, needBytes, m_stream);
        }
    }

    void Simulator::releaseRenderHalfExternal() {
        if (m_extRenderHalf) {
            cudaDestroyExternalMemory(m_extRenderHalf);
            m_extRenderHalf = nullptr;
        }
        m_renderHalfMappedPtr = nullptr;
        m_renderHalfBytes = 0;
    }

    // ===== 追加：调试插桩（仅非 Graph 模式） =====
    static void DebugPrintParticle0(const char* tag,
        const DeviceBuffers& bufs,
        uint32_t N,
        float dt) {
        if (N == 0) { std::fprintf(stderr, "[IntVelDbg][%s] N=0\n", tag); return; }
        float4 pos0{}, next0{}, vel0{}, delta0{};
        if (bufs.d_pos_curr) cudaMemcpy(&pos0, bufs.d_pos_curr, sizeof(float4), cudaMemcpyDeviceToHost);
        if (bufs.d_pos_next) cudaMemcpy(&next0, bufs.d_pos_next, sizeof(float4), cudaMemcpyDeviceToHost);
        if (bufs.d_vel) cudaMemcpy(&vel0, bufs.d_vel, sizeof(float4), cudaMemcpyDeviceToHost);
        if (bufs.d_delta) cudaMemcpy(&delta0, bufs.d_delta, sizeof(float4), cudaMemcpyDeviceToHost);
        float dx = next0.x - pos0.x; float dy = next0.y - pos0.y; float dz = next0.z - pos0.z;
        std::fprintf(stderr,
            "[IntVelDbg][%s] curr=(%.6f,%.6f,%.6f) next=(%.6f,%.6f,%.6f) d=(%.6e,%.6e,%.6e) vel=(%.6f,%.6f,%.6f) delta(xsph)=(%.6f,%.6f,%.6f) d/dt=(%.6f,%.6f,%.6f) dt=%.6f\n",
            tag, pos0.x, pos0.y, pos0.z, next0.x, next0.y, next0.z, dx, dy, dz, vel0.x, vel0.y, vel0.z, delta0.x, delta0.y, delta0.z, dx / dt, dy / dt, dz / dt, dt);
    }
    static void DebugPrintVel0(const char* tag, const DeviceBuffers& bufs, uint32_t N, float dt) { if (N == 0 || !bufs.d_vel || !bufs.d_pos_curr || !bufs.d_pos_next) return; float4 v0{}, c0{}, n0{}; cudaMemcpy(&v0, bufs.d_vel, sizeof(float4), cudaMemcpyDeviceToHost); cudaMemcpy(&c0, bufs.d_pos_curr, sizeof(float4), cudaMemcpyDeviceToHost); cudaMemcpy(&n0, bufs.d_pos_next, sizeof(float4), cudaMemcpyDeviceToHost); float dx = n0.x - c0.x, dy = n0.y - c0.y, dz = n0.z - c0.z; std::fprintf(stderr, "[PP-Debug][%s][Vel0] v=(%.6f,%.6f,%.6f) Δ=(%.6f,%.6f,%.6f) Δ/dt=(%.6f,%.6f,%.6f) dt=%.6f\n", tag, v0.x, v0.y, v0.z, dx, dy, dz, dx / dt, dy / dt, dz / dt, dt); }
    static void DebugPrintPosDelta(const char* tag, const DeviceBuffers& bufs, uint32_t N) { if (N == 0 || !bufs.d_pos_curr || !bufs.d_pos_next) { std::fprintf(stderr, "[PP-Debug][%s] N=0 or null\n", tag); return; } float4 a{}, b{}; cudaMemcpy(&a, bufs.d_pos_curr, sizeof(float4), cudaMemcpyDeviceToHost); cudaMemcpy(&b, bufs.d_pos_next, sizeof(float4), cudaMemcpyDeviceToHost); }
    static void DebugPrintP0(const char* tag, const DeviceBuffers& bufs, uint32_t N, float dt) { if (N == 0) { std::fprintf(stderr, "[P0][%s] N=0\n", tag); return; } float4 c{}, n{}, v{}, d{}; if (bufs.d_pos_curr) cudaMemcpy(&c, bufs.d_pos_curr, sizeof(float4), cudaMemcpyDeviceToHost); if (bufs.d_pos_next) cudaMemcpy(&n, bufs.d_pos_next, sizeof(float4), cudaMemcpyDeviceToHost); if (bufs.d_vel) cudaMemcpy(&v, bufs.d_vel, sizeof(float4), cudaMemcpyDeviceToHost); if (bufs.d_delta) cudaMemcpy(&d, bufs.d_delta, sizeof(float4), cudaMemcpyDeviceToHost); float dx = n.x - c.x, dy = n.y - c.y, dz = n.z - c.z; std::fprintf(stderr, "[P0][%s] curr=(%.6f,%.6f,%.6f) next=(%.6f,%.6f,%.6f) Δ=(%.6e,%.6e,%.6e) vel=(%.6f,%.6f,%.6f) delta=(%.6f,%.6f,%.6f) Δ/dt=(%.6f,%.6f,%.6f) dt=%.6f\n", tag, c.x, c.y, c.z, n.x, n.y, n.z, dx, dy, dz, v.x, v.y, v.z, d.x, d.y, d.z, dx / dt, dy / dt, dz / dt, dt); }

    // ===== Initialization / Shutdown =====
    bool Simulator::initialize(const SimParams& p) {
        prof::Range rInit("Sim.Initialize", prof::Color(0x10, 0x90, 0xF0));

        m_params = p;
        CUDA_CHECK(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));

        const auto& c = console::Instance();
        m_frameTimingEveryN = c.perf.frame_timing_every_n;
        m_frameTimingEnabled = (m_frameTimingEveryN != 0);
        m_useHashedGrid = p.useHashedGrid;

        // 上下文
        m_ctx.bufs = &m_bufs;
        m_ctx.grid = &m_grid;
        m_ctx.useHashedGrid = m_useHashedGrid;
        if (m_useHashedGrid) m_gridStrategy = std::make_unique<HashedGridStrategy>();
        else                 m_gridStrategy = std::make_unique<DenseGridStrategy>();
        m_ctx.gridStrategy = m_gridStrategy.get();
        m_ctx.dispatcher = &m_kernelDispatcher;

        uint32_t capacity = (p.maxParticles > 0) ? p.maxParticles : p.numParticles;
        if (capacity == 0) capacity = 1;

        bool needHalf = (p.precision.positionStore == NumericType::FP16_Packed) ||
            (p.precision.velocityStore == NumericType::FP16_Packed) ||
            (p.precision.lambdaStore == NumericType::FP16) ||
            (p.precision.densityStore == NumericType::FP16) ||
            (p.precision.auxStore == NumericType::FP16_Packed || p.precision.auxStore == NumericType::FP16);

        if (needHalf) m_bufs.allocateWithPrecision(p.precision, capacity);
        else          m_bufs.allocate(capacity);

        UpdateDevicePrecisionView(m_bufs, p.precision);

        m_grid.allocateIndices(capacity);
        m_grid.ensureCompactCapacity(capacity);
        if (!buildGrid(m_params)) return false;

        if (capacity > 0) {
            std::vector<uint32_t> h_idx(capacity);
            for (uint32_t i = 0; i < capacity; ++i) h_idx[i] = i;
            CUDA_CHECK(cudaMemcpy(m_grid.d_indices, h_idx.data(),
                sizeof(uint32_t) * capacity, cudaMemcpyHostToDevice));
        }

        if (p.numParticles > 0) {
            CUDA_CHECK(cudaMemcpy(m_bufs.d_pos_next, m_bufs.d_pos_curr,
                sizeof(float4) * p.numParticles,
                cudaMemcpyDeviceToDevice));
            if (needHalf) m_bufs.packAllToHalf(p.numParticles, m_stream);
        }
        UploadSimPosTableConst(m_bufs.d_pos_curr, m_bufs.d_pos_next);

        m_canPingPongPos = true;
        m_graphDirty = true;
        m_captured = {};
        m_cachedNodesReady = false;
        m_nodeRecycleFull = nullptr;
        m_lastFrameMs = -1.0f;
        m_evCursor = 0;
        m_lastParamUpdateFrame = -1;
        m_swappedThisFrame = false;

        if (m_pipeline.phases().empty()) {
            BuildDefaultPipelines(m_pipeline);
            PostOpsConfig postCfg{};
            postCfg.enableXsph = (p.xsph_c > 0.f);
            postCfg.enableBoundary = true;
            postCfg.enableRecycle = true;
            m_pipeline.post().configure(postCfg, m_useHashedGrid, postCfg.enableXsph);
        }

        m_paramTracker.capture(m_params, m_numCells);
        BindDeviceGlobalsFrom(m_bufs);

        std::fprintf(stderr,
            "[Init][PingPong] curr=%p next=%p N=%u\n",
            (void*)m_bufs.d_pos_curr, (void*)m_bufs.d_pos_next, p.numParticles);
        return true;
    }

    void Simulator::shutdown() {
        if (m_extRenderHalf) {
            cudaDestroyExternalMemory(m_extRenderHalf);
            m_extRenderHalf = nullptr;
        }

        m_grid.releaseAll();

        if (m_graphExecFull) { cudaGraphExecDestroy(m_graphExecFull);  m_graphExecFull = nullptr; }
        if (m_graphFull) { cudaGraphDestroy(m_graphFull);          m_graphFull = nullptr; }

        for (int i = 0; i < 2; ++i) {
            if (m_evStart[i]) { cudaEventDestroy(m_evStart[i]); m_evStart[i] = nullptr; }
            if (m_evEnd[i]) { cudaEventDestroy(m_evEnd[i]);   m_evEnd[i] = nullptr; }
        }

        if (m_stream) { cudaStreamDestroy(m_stream); m_stream = nullptr; }
    }

    // ===== 网格 & 参数 =====
    bool Simulator::buildGrid(const SimParams& p) {
        int3 dim = GridSystem::ComputeDims(p.grid);
        m_numCells = GridSystem::NumCells(dim);
        if (m_numCells == 0) return false;
        m_grid.allocGridRanges(m_numCells);
        m_params.grid = p.grid;
        m_params.grid.dim = dim;
        return true;
    }

    bool Simulator::updateGridIfNeeded(const SimParams& p) {
        int3 dim = GridSystem::ComputeDims(p.grid);
        uint32_t newNC = GridSystem::NumCells(dim);
        bool changed = false;

        if (newNC != m_numCells) changed = true;
        if (!gridEqual(p.grid, m_captured.grid)) changed = true;

        if (changed) {
            m_grid.resizeGridRanges(newNC);
            m_numCells = newNC;
            m_graphDirty = true;
        }

        m_params.grid = p.grid;
        m_params.grid.dim = dim;
        m_grid.ensureCompactCapacity(p.maxParticles > 0 ? p.maxParticles : p.numParticles);
        return changed;
    }

    bool Simulator::ensureSortTemp(std::size_t bytes) {
        m_grid.ensureSortTemp(bytes);
        return true;
    }

    // ===== Graph Node 缓存 =====
    bool Simulator::cacheGraphNodes() {
        // 仅记录 recycle 节点；不再解析 kernelParams 指针内容
        m_nodeRecycleFull = nullptr;
        m_kpRecycleBaseFull = {};
        m_cachedNodesReady = false;

        // 迁移后不再需要 velocity 节点列表
        m_velNodesFull.clear();
        m_velNodeParamsBaseFull.clear();
        m_cachedVelNodes = true; // 标记为已“缓存”，避免其它地方重复工作

        void* target = GetRecycleKernelPtr();
        if (!m_graphFull) {
            m_cachedNodesReady = true;
            return true;
        }

        size_t n = 0;
        cudaError_t err = cudaGraphGetNodes(m_graphFull, nullptr, &n);
        if (err != cudaSuccess || n == 0) {
            m_cachedNodesReady = true;
            return true;
        }

        std::vector<cudaGraphNode_t> nodes(n);
        CUDA_CHECK(cudaGraphGetNodes(m_graphFull, nodes.data(), &n));

        for (auto nd : nodes) {
            cudaGraphNodeType t;
            if (cudaGraphNodeGetType(nd, &t) != cudaSuccess) continue;
            if (t != cudaGraphNodeTypeKernel) continue;

            cudaKernelNodeParams kp{};
            if (cudaGraphKernelNodeGetParams(nd, &kp) != cudaSuccess) continue;

            // 仅匹配 recycle kernel
            if (kp.func == target) {
                m_nodeRecycleFull = nd;
                m_kpRecycleBaseFull = kp;
                break; // 找到即退出
            }
        }

        m_cachedNodesReady = true;
        return true;
    }

    // ===== Graph 变化检测 =====
    bool Simulator::structuralGraphChange(const SimParams& p) const {
        if (!m_graphExecFull) return true;
        if (p.solverIters != m_captured.solverIters)   return true;
        if (p.maxNeighbors != m_captured.maxNeighbors)  return true;
        if (p.numParticles != m_captured.numParticles)  return true;
        int3 dim = GridSystem::ComputeDims(p.grid);
        uint32_t nc = GridSystem::NumCells(dim);
        if (nc != m_captured.numCells) return true;
        if (!gridEqual(p.grid, m_captured.grid)) return true;
        return false;
    }

    bool Simulator::paramOnlyGraphChange(const SimParams& p) const {
        if (structuralGraphChange(p)) return false;
        float dtRel = fabsf(p.dt - m_captured.dt) / fmaxf(1e-9f, fmaxf(p.dt, m_captured.dt));
        if (dtRel < 0.002f &&
            approxEq3(p.gravity, m_captured.gravity) &&
            approxEq(p.restDensity, m_captured.restDensity) &&
            kernelEqualRelaxed(p.kernel, m_captured.kernel)) {
            return false;
        }
        if (!approxEq(p.dt, m_captured.dt))                return true;
        if (!approxEq3(p.gravity, m_captured.gravity))     return true;
        if (!approxEq(p.restDensity, m_captured.restDensity)) return true;
        if (!kernelEqualRelaxed(p.kernel, m_captured.kernel)) return true;
        return false;
    }

    // ===== Graph 构建与参数更新 =====
    bool Simulator::captureGraphIfNeeded(const SimParams& p) {
        if (!m_graphDirty) return true;

        if (!m_graphPointersChecked) {
            if (!m_bufs.d_pos_curr || !m_bufs.d_vel || !m_bufs.d_pos_next) {
                std::fprintf(stderr,
                    "[Graph][Error] Required buffers null before capture (curr=%p vel=%p next=%p)\n",
                    (void*)m_bufs.d_pos_curr, (void*)m_bufs.d_vel, (void*)m_bufs.d_pos_next);
            }
            m_graphPointersChecked = true;
        }

        prof::Range r("Sim.GraphCapture", prof::Color(0xE0,0x60,0x20));
        GraphBuilder builder;
        auto result = builder.BuildStructural(*this, p);
        return result.structuralRebuilt && result.reuseSucceeded;
    }

    bool Simulator::updateGraphsParams(const SimParams& p) {
        if (!m_paramDirty) return true;
        if (!m_graphExecFull) return false;

        const auto& c = console::Instance();
        prof::Range r("Sim.GraphUpdate", prof::Color(0xA0, 0x50, 0x10));
        GraphBuilder builder;
        auto res = builder.UpdateDynamic(*this, p,
            c.perf.graph_param_update_min_interval,
            m_frameIndex,
            m_lastParamUpdateFrame);

        if (m_graphDirty) {
            auto rs = builder.BuildStructural(*this, p);
            if (!rs.structuralRebuilt) return false;
            m_lastParamUpdateFrame = m_frameIndex;
            return true;
        }

        if (res.dynamicUpdated) {
            m_lastParamUpdateFrame = m_frameIndex;
            return true;
        }
        return !m_paramDirty;
    }

    // ===== 阶段 kernels =====
    void Simulator::kHashKeys(cudaStream_t s, const SimParams& p) {
        prof::Range r("Phase.HashKeys", prof::Color(0x60,0xB0,0x40));
        bool useMP = UseHalfForPosition(p, Stage::GridBuild, m_bufs);
        if (useMP)
            LaunchHashKeysMP(m_grid.d_cellKeys, m_grid.d_indices,
                             m_bufs.d_pos_next, m_bufs.d_pos_next_h4,
                             p.grid, p.numParticles, s);
        else
            LaunchHashKeys(m_grid.d_cellKeys, m_grid.d_indices,
                           m_bufs.d_pos_next, p.grid, p.numParticles, s);
    }

    void Simulator::kSort(cudaStream_t s, const SimParams& p) {
        prof::Range r("Phase.SortPairs", prof::Color(0x90, 0x50, 0xF0));
        LaunchSortPairs(m_grid.d_sortTemp, m_grid.sortTempBytes,
                        m_grid.d_cellKeys, m_grid.d_cellKeys_sorted,
                        m_grid.d_indices, m_grid.d_indices_sorted,
                        p.numParticles, s);
    }

    void Simulator::kCellRanges(cudaStream_t s, const SimParams& p) {
        prof::Range r("Phase.CellRanges.Dense", prof::Color(0x40, 0xD0, 0xD0));
        CUDA_CHECK(cudaMemsetAsync(m_grid.d_cellStart, 0xFF, sizeof(uint32_t) * m_numCells, s));
        CUDA_CHECK(cudaMemsetAsync(m_grid.d_cellEnd, 0xFF, sizeof(uint32_t) * m_numCells, s));
        LaunchCellRanges(m_grid.d_cellStart, m_grid.d_cellEnd,
                         m_grid.d_cellKeys_sorted, p.numParticles, m_numCells, s);
    }

    void Simulator::buildKeyToCompactMap(uint32_t compactCountHost) {
        if (!m_grid.d_keyToCompact || m_numCells ==0) return;
        // If no entries, just memset to0xFF
        if (compactCountHost ==0) {
            CUDA_CHECK(cudaMemset(m_grid.d_keyToCompact,0xFF, sizeof(uint32_t) * m_numCells));
            return;
        }
        std::vector<uint32_t> hKeys(compactCountHost);
        CUDA_CHECK(cudaMemcpy(hKeys.data(), m_grid.d_cellUniqueKeys,
                              sizeof(uint32_t) * compactCountHost, cudaMemcpyDeviceToHost));
        std::vector<uint32_t> hMap(m_numCells,0xFFFFFFFFu);
        for (uint32_t i =0; i < compactCountHost; ++i) {
            uint32_t key = hKeys[i];
            if (key < m_numCells) hMap[key] = i;
        }
        CUDA_CHECK(cudaMemcpy(m_grid.d_keyToCompact, hMap.data(), sizeof(uint32_t) * m_numCells, cudaMemcpyHostToDevice));
    }

    void Simulator::kCellRangesCompact(cudaStream_t s, const SimParams& p) {
        // 控制重建：按照 p.compactRebuildEveryN 与相对粒子数变化阈值 m_compactRebuildParticleRelThreshold
        int rebuildEvery = (p.compactRebuildEveryN <1 ?1 : p.compactRebuildEveryN);
        bool rebuild = true;
        if (m_lastCompactRebuildFrame >=0) {
            int frameDelta = m_frameIndex - m_lastCompactRebuildFrame;
            double relChange =0.0;
            if (m_lastCompactParticleCount >0) {
                relChange = fabs(double(p.numParticles) - double(m_lastCompactParticleCount)) / double(m_lastCompactParticleCount);
            }
            // 若未到帧间隔且粒子数相对变化低于阈值则跳过重建
            if (frameDelta < rebuildEvery && relChange < m_compactRebuildParticleRelThreshold) rebuild = false;
            if (!rebuild && p.logGridCompactStats && console::Instance().debug.printDiagnostics) {
                std::fprintf(stderr, "[Compact][Skip] frame=%d delta=%d every=%d relChange=%.3f thr=%.3f mode=%s\n", m_frameIndex, frameDelta, rebuildEvery, relChange, m_compactRebuildParticleRelThreshold, p.compactBinarySearch?"binary":"hash");
            }
        }
        if (!rebuild) return;

        prof::Range r("Phase.CellRanges.Compact", prof::Color(0x40,0xD0,0xD0));
        LaunchCellRangesCompact(m_grid.d_cellUniqueKeys, m_grid.d_cellOffsets, m_grid.d_compactCount,
                                m_grid.d_cellKeys_sorted, p.numParticles, s);
        // 更新跟踪状态
        m_lastCompactRebuildFrame = m_frameIndex;
        m_lastCompactParticleCount = p.numParticles;
        m_lastCompactUsedBinary = p.compactBinarySearch;

        if (p.useHashedGrid) {
            if (!p.compactBinarySearch) {
                // 构建 key -> compact 段索引映射以供 hashLookupKey 使用
                uint32_t hCount =0; CUDA_CHECK(cudaMemcpy(&hCount, m_grid.d_compactCount, sizeof(uint32_t), cudaMemcpyDeviceToHost));
                buildKeyToCompactMap(hCount);
            } else if (m_grid.d_keyToCompact) {
                // 二分模式：清空映射避免被误用
                CUDA_CHECK(cudaMemsetAsync(m_grid.d_keyToCompact,0xFF, sizeof(uint32_t) * m_numCells, s));
            }
        }

        if (p.logGridCompactStats && console::Instance().debug.printDiagnostics) {
            uint32_t hCount =0; CUDA_CHECK(cudaMemcpy(&hCount, m_grid.d_compactCount, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            if (hCount >0) {
                std::vector<uint32_t> hOffsets(hCount +1);
                CUDA_CHECK(cudaMemcpy(hOffsets.data(), m_grid.d_cellOffsets, sizeof(uint32_t) * (hCount +1), cudaMemcpyDeviceToHost));
                uint64_t sumOcc =0; uint32_t maxOcc =0; uint32_t minOcc = 0xFFFFFFFFu;
                for (uint32_t i =0; i < hCount; ++i) {
                    uint32_t occ = hOffsets[i+1] - hOffsets[i];
                    sumOcc += occ;
                    maxOcc = (occ > maxOcc ? occ : maxOcc);
                    minOcc = (occ < minOcc ? occ : minOcc);
                }
                double avgOcc = double(sumOcc) / double(hCount);
                if (minOcc == 0xFFFFFFFFu) minOcc = 0;
                std::fprintf(stderr, "[Compact][Stats] frame=%d cells=%u occAvg=%.2f occMax=%u occMin=%u mode=%s\n", m_frameIndex, hCount, avgOcc, maxOcc, minOcc, p.compactBinarySearch?"binary":"hash");
            } else {
                std::fprintf(stderr, "[Compact][Stats] frame=%d cells=0 (empty) mode=%s\n", m_frameIndex, p.compactBinarySearch?"binary":"hash");
            }
        }
    }

    void Simulator::kSolveIter(cudaStream_t s, const SimParams& p) {
        prof::Range r("Phase.SolveIter", prof::Color(0xE0, 0x80, 0x40));
        DeviceParams dp = MakeDeviceParams(p);
        for (int iter = 0; iter < p.solverIters; ++iter) {
            if (m_useHashedGrid) {
                LaunchLambdaCompactGlobals(m_grid.d_indices_sorted,
                    m_grid.d_cellKeys_sorted,
                    m_grid.d_cellUniqueKeys,
                    m_grid.d_cellOffsets,
                    m_grid.d_compactCount,
                    dp, p.numParticles, s);
                LaunchDeltaApplyCompactGlobals(m_grid.d_indices_sorted,
                    m_grid.d_cellKeys_sorted,
                    m_grid.d_cellUniqueKeys,
                    m_grid.d_cellOffsets,
                    m_grid.d_compactCount,
                    dp, p.numParticles, s);
            }
            else {
                LaunchLambdaDenseGlobals(m_grid.d_indices_sorted,
                    m_grid.d_cellKeys_sorted,
                    m_grid.d_cellStart,
                    m_grid.d_cellEnd,
                    dp, p.numParticles, s);
                LaunchDeltaApplyDenseGlobals(m_grid.d_indices_sorted,
                    m_grid.d_cellKeys_sorted,
                    m_grid.d_cellStart,
                    m_grid.d_cellEnd,
                    dp, p.numParticles, s);
            }
            // 迭代内仍用无弹性边界校正（不使用 XSPH）
            LaunchBoundaryGlobals(p.grid, 0.0f, p.numParticles, false, s);
        }
    }

    void Simulator::kVelocityAndPost(cudaStream_t s, const SimParams& p) {
        prof::Range r("Phase.VelocityPost", prof::Color(0xC0, 0x40, 0xA0));

        // 1. 根据当前 pos / pos_pred 计算速度
        LaunchVelocityGlobals(1.0f / p.dt, p.numParticles, s);

        // 2. 可选 XSPH 平滑（写入 g_delta）
        bool xsphApplied = false;
        if (p.xsph_c > 0.f && p.numParticles > 0) {
            DeviceParams dp = MakeDeviceParams(p);
            if (m_useHashedGrid) {
                LaunchXsphCompactGlobals(m_grid.d_indices_sorted,
                    m_grid.d_cellKeys_sorted,
                    m_grid.d_cellUniqueKeys,
                    m_grid.d_cellOffsets,
                    m_grid.d_compactCount,
                    dp, p.numParticles, s);
            }
            else {
                LaunchXsphDenseGlobals(m_grid.d_indices_sorted,
                    m_grid.d_cellKeys_sorted,
                    m_grid.d_cellStart,
                    m_grid.d_cellEnd,
                    dp, p.numParticles, s);
            }
            xsphApplied = true;
        }

        // 3. 边界：仅一次（带 restitution）；如果 XSPH 已应用，BoundaryGlobals 会读取 g_delta 并提交到 g_vel
        LaunchBoundaryGlobals(p.grid, p.boundaryRestitution, p.numParticles, xsphApplied, s);

        // 4. Recycle 基于最终有效速度（若 xsphApplied 为 true 则已经在 boundary 时写入 g_vel）
        float4* effectiveVel = m_bufs.d_vel; // Boundary 已提交最终速度
        LaunchRecycleToNozzleConst(m_bufs.d_pos_curr,
                                   m_bufs.d_pos_next,
                                   effectiveVel,
                                   p.grid, p.dt, p.numParticles, 0, s);

        cudaStreamSynchronize(s);
    }

    void Simulator::kIntegratePred(cudaStream_t s, const SimParams& p) {
        prof::Range r("Phase.Integrate", prof::Color(0x50, 0xA0, 0xFF));
        LaunchIntegratePredGlobals(p.gravity, p.dt, p.numParticles, s);
        cudaStreamSynchronize(s);
    }

    // ===== Step =====
    bool Simulator::step(const SimParams& p) {
        prof::Range rf("Sim.Step", prof::Color(0x30, 0x30, 0xA0));
        g_simFrameIndex = m_frameIndex;
        m_swappedThisFrame = false;

        m_params = p;

        bool expected = m_canPingPongPos;
        if (m_ctx.pingPongPos != expected)
            m_ctx.pingPongPos = expected;

        const auto& cHot = console::Instance();
        if (m_frameTimingEveryN != cHot.perf.frame_timing_every_n) {
            m_frameTimingEveryN = cHot.perf.frame_timing_every_n;
            m_frameTimingEnabled = (m_frameTimingEveryN != 0);
        }

        EmitParams ep{};
        console::BuildEmitParams(console::Instance(), ep);
        uint32_t capacity = m_bufs.capacity;
        if (m_params.maxParticles == 0) m_params.maxParticles = capacity;
        m_params.maxParticles = std::min<uint32_t>(m_params.maxParticles, capacity);

        uint32_t emitted = Emitter::EmitFaucet(m_bufs, m_params,
            console::Instance(), ep, m_frameIndex, m_stream);

        if (emitted > 0 && m_bufs.anyHalf())
            m_bufs.packAllToHalf(m_params.numParticles, m_stream);

        if (m_params.numParticles > m_bufs.capacity) {
            const auto& pr = m_params.precision;
            bool needHalf2 = (pr.positionStore == NumericType::FP16_Packed ||
                pr.velocityStore == NumericType::FP16_Packed ||
                pr.lambdaStore == NumericType::FP16 ||
                pr.densityStore == NumericType::FP16 ||
                pr.auxStore == NumericType::FP16_Packed || pr.auxStore == NumericType::FP16);
            if (needHalf2) m_bufs.allocateWithPrecision(pr, m_params.numParticles);
            else m_bufs.allocate(m_params.numParticles);

            UpdateDevicePrecisionView(m_bufs, m_params.precision);

            m_grid.allocateIndices(m_bufs.capacity);
            std::vector<uint32_t> h_idx(m_bufs.capacity);
            for (uint32_t i = 0; i < m_bufs.capacity; ++i) h_idx[i] = i;
            CUDA_CHECK(cudaMemcpy(m_grid.d_indices, h_idx.data(),
                sizeof(uint32_t) * m_bufs.capacity,
                cudaMemcpyHostToDevice));
            m_graphDirty = true;
            UploadSimPosTableConst(m_bufs.d_pos_curr, m_bufs.d_pos_next);
        }

        SetEmitParamsAsync(&ep, m_stream);
        updateGridIfNeeded(m_params);

        bool structuralChanged = m_paramTracker.structuralChanged(m_params, m_numCells);
        bool dynamicChanged = (!structuralChanged) && m_paramTracker.dynamicChanged(m_params);
        if (structuralChanged) m_graphDirty = true;
        else if (dynamicChanged) m_paramDirty = true;

        if (m_graphDirty)
            captureGraphIfNeeded(m_params);
        else if (m_paramDirty)
            updateGraphsParams(m_params);

        bool doTiming = m_frameTimingEnabled &&
            (m_frameTimingEveryN <= 1 ||
                (m_frameIndex % m_frameTimingEveryN) == 0);

        int cur = (m_evCursor & 1);
        int prev = ((m_evCursor + 1) & 1);
        if (doTiming) {
            for (int i = 0; i < 2; ++i) {
                if (!m_evStart[i]) cudaEventCreate(&m_evStart[i]);
                if (!m_evEnd[i])   cudaEventCreate(&m_evEnd[i]);
            }
            if (m_evStart[cur] && m_evEnd[cur])
                CUDA_CHECK(cudaEventRecord(m_evStart[cur], m_stream));
        }

        auto updNode = [&](cudaGraphExec_t exec,
            cudaGraphNode_t& node,
            cudaKernelNodeParams& base) {
                if (!exec || !node || m_params.numParticles == 0) return;
                uint32_t blocks = (m_params.numParticles + 255u) / 256u;
                if (blocks == 0) blocks = 1;
                cudaKernelNodeParams kp = base;
                kp.gridDim = dim3(blocks, 1, 1);
                cudaError_t err = cudaGraphExecKernelNodeSetParams(exec, node, &kp);
                if (err == cudaErrorInvalidResourceHandle) {
                    cacheGraphNodes();
                    node = m_nodeRecycleFull;
                    base = m_kpRecycleBaseFull;
                    if (node) {
                        kp = base; kp.gridDim = dim3(blocks, 1, 1);
                        cudaError_t err2 = cudaGraphExecKernelNodeSetParams(exec, node, &kp);
                        if (err2 != cudaSuccess)
                            std::fprintf(stderr,
                                "[Graph][Error] Retry setParams failed err=%d (%s)\n",
                                (int)err2, cudaGetErrorString(err2));
                    }
                }
            };

        if (!m_graphExecFull) {
            if (console::Instance().debug.printErrors)
                std::fprintf(stderr, "[Step][Error] Graph not ready.\n");
            return false;
        }

        updNode(m_graphExecFull, m_nodeRecycleFull, m_kpRecycleBaseFull);
        CUDA_CHECK(cudaGraphLaunch(m_graphExecFull, m_stream));

        performPingPongSwap(m_params.numParticles);

        signalSimFence();
        if (p.ghostParticleCount != 0) {
            UploadGhostCount(p.ghostParticleCount);
        }

        if (doTiming) {
            if (m_evEnd[cur]) CUDA_CHECK(cudaEventRecord(m_evEnd[cur], m_stream));
            if (m_evCursor > 0 && m_evStart[prev] && m_evEnd[prev]) {
                if (cudaEventQuery(m_evEnd[prev]) == cudaSuccess) {
                    float ms = 0.f;
                    CUDA_CHECK(cudaEventElapsedTime(&ms, m_evStart[prev], m_evEnd[prev]));
                    m_lastFrameMs = ms;
                }
            }
        }

        ++m_frameIndex;
        if (p.precision.renderTransfer == NumericType::FP16_Packed ||
            p.precision.renderTransfer == NumericType::FP16) {
            publishRenderHalf(m_params.numParticles);
        }
        return true;
    }

    void Simulator::seedBoxLatticeAuto(uint32_t total, float3 origin, float spacing) {
        float3 size = make_float3(
            fmaxf(0.f, m_params.grid.maxs.x - m_params.grid.mins.x),
            fmaxf(0.f, m_params.grid.maxs.y - m_params.grid.mins.y),
            fmaxf(0.f, m_params.grid.maxs.z - m_params.grid.mins.z));

        auto cap = [&spacing](float len) -> uint32_t { return (spacing > 0.f) ? (uint32_t)floorf(len / spacing) : 0u; };

        uint32_t maxX = std::max<uint32_t>(1, cap(size.x));
        uint32_t maxY = std::max<uint32_t>(1, cap(size.y));
        uint32_t maxZ = std::max<uint32_t>(1, cap(size.z));

        uint64_t T = std::max<uint32_t>(1u, total);
        uint32_t nx = std::min<uint32_t>(maxX, (uint32_t)ceil(pow(double(T), 1.0 / 3.0)));
        uint64_t remXY = (T + nx - 1) / nx;
        uint32_t ny = std::min<uint32_t>(maxY, (uint32_t)ceil(sqrt((double)remXY)));
        uint32_t nz = std::min<uint32_t>(maxZ, (uint32_t)ceil(double(remXY) / std::max<uint32_t>(1u, ny)));

        auto safeMul = [](uint64_t a, uint64_t b) {
            return (a > 0 && b > (UINT64_MAX / a)) ? UINT64_MAX : a * b;
            };
        while (safeMul(nx, safeMul(ny, nz)) < T) {
            if (nx < maxX)      ++nx;
            else if (ny < maxY) ++ny;
            else if (nz < maxZ) ++nz;
            else break;
        }

        uint64_t nreq64 = (uint64_t)nx * ny * nz;
        uint32_t Nreq = (nreq64 > UINT32_MAX) ? UINT32_MAX : (uint32_t)nreq64;
        if (Nreq < total) {
            const auto& c = console::Instance();
            if (c.debug.printWarnings)
                std::fprintf(stderr, "[Warn] seedBoxLatticeAuto: Nreq(%u) < total(%u). Potential overlap.\n", Nreq, total);
        }

        float3 margin = make_float3(spacing * 0.25f, spacing * 0.25f, spacing * 0.25f);
        origin.x = fminf(fmaxf(origin.x, m_params.grid.mins.x + margin.x), m_params.grid.maxs.x - margin.x);
        origin.y = fminf(fmaxf(origin.y, m_params.grid.mins.y + margin.y), m_params.grid.maxs.y - margin.y);
        origin.z = fminf(fmaxf(origin.z, m_params.grid.mins.z + margin.z), m_params.grid.maxs.z - margin.z);

        seedBoxLattice(nx, ny, nz, origin, spacing);
    }

    void Simulator::signalSimFence() {
        if (!m_extTimelineSem) return;
        ++m_simFenceValue;
        cudaExternalSemaphoreSignalParams params{};
        params.params.fence.value = m_simFenceValue;
        params.flags = 0;
        cudaSignalExternalSemaphoresAsync(&m_extTimelineSem, &params, 1, m_stream);
    }

    bool Simulator::enableExternalPingPong(void* sharedHandleA, size_t bytesA,
        void* sharedHandleB, size_t bytesB)
    {
        if (!sharedHandleA || !sharedHandleB || bytesA == 0 || bytesB == 0) {
            std::fprintf(stderr, "[PingPong][Error] invalid handles or sizes A=%p B=%p bytesA=%zu bytesB=%zu\n",
                sharedHandleA, sharedHandleB, bytesA, bytesB);
            return false;
        }
        if (bytesA != bytesB) {
            std::fprintf(stderr, "[PingPong][Warn] size mismatch bytesA=%zu bytesB=%zu (using bytesA)\n", bytesA, bytesB);
        }
        if (bytesA % sizeof(float4) != 0) {
            std::fprintf(stderr, "[PingPong][Error] bytes (%zu) not multiple of sizeof(float4)\n", bytesA);
            return false;
        }

        // 清理旧外部内存
        if (m_extPosPred) { cudaDestroyExternalMemory(m_extPosPred); m_extPosPred = nullptr; }
        if (m_extraExternalMemB) { cudaDestroyExternalMemory(m_extraExternalMemB); m_extraExternalMemB = nullptr; }

        // 导入 A
        cudaExternalMemoryHandleDesc descA{}; descA.type = cudaExternalMemoryHandleTypeD3D12Resource;
        descA.handle.win32.handle = (HANDLE)sharedHandleA; descA.size = bytesA; descA.flags = cudaExternalMemoryDedicated;
        if (cudaImportExternalMemory(&m_extPosPred, &descA) != cudaSuccess) {
            std::fprintf(stderr, "[PingPong][Error] import ext A failed\n");
            m_extPosPred = nullptr; return false;
        }

        // 导入 B
        cudaExternalMemoryHandleDesc descB{}; descB.type = cudaExternalMemoryHandleTypeD3D12Resource;
        descB.handle.win32.handle = (HANDLE)sharedHandleB; descB.size = bytesA; descB.flags = cudaExternalMemoryDedicated;
        if (cudaImportExternalMemory(&m_extraExternalMemB, &descB) != cudaSuccess) {
            std::fprintf(stderr, "[PingPong][Error] import ext B failed\n");
            cudaDestroyExternalMemory(m_extPosPred); m_extPosPred = nullptr;
            m_extraExternalMemB = nullptr; return false;
        }

        // 映射
        cudaExternalMemoryBufferDesc bufDesc{}; bufDesc.offset = 0; bufDesc.size = bytesA;
        void* devPtrA = nullptr;
        if (cudaExternalMemoryGetMappedBuffer(&devPtrA, m_extPosPred, &bufDesc) != cudaSuccess) {
            std::fprintf(stderr, "[PingPong][Error] map A failed\n");
            cudaDestroyExternalMemory(m_extPosPred); m_extPosPred = nullptr;
            cudaDestroyExternalMemory(m_extraExternalMemB); m_extraExternalMemB = nullptr;
            return false;
        }
        void* devPtrB = nullptr;
        if (cudaExternalMemoryGetMappedBuffer(&devPtrB, m_extraExternalMemB, &bufDesc) != cudaSuccess) {
            std::fprintf(stderr, "[PingPong][Error] map B failed\n");
            cudaDestroyExternalMemory(m_extPosPred); m_extPosPred = nullptr;
            cudaDestroyExternalMemory(m_extraExternalMemB); m_extraExternalMemB = nullptr;
            return false;
        }

        uint32_t capacity = static_cast<uint32_t>(bytesA / sizeof(float4));
        m_bufs.adoptExternalPingPong(reinterpret_cast<float4*>(devPtrA),
            reinterpret_cast<float4*>(devPtrB),
            capacity);
        UploadSimPosTableConst(m_bufs.d_pos_curr, m_bufs.d_pos_next);
        m_canPingPongPos = true;

        std::fprintf(stderr, "[PingPong][Ready] A=%p B=%p capacity(float4)=%u\n",
            devPtrA, devPtrB, capacity);
        return true;
    }
 
    void Simulator::performPingPongSwap(uint32_t N) {
        if (!m_ctx.pingPongPos || N == 0) return;
        if (m_bufs.isExternalPingPong()) {
            m_bufs.swapPingPongPositions();
            m_swappedThisFrame = true;
            UploadSimPosTableConst(m_bufs.posCurr(), m_bufs.posNext());
            BindDeviceGlobalsFrom(m_bufs); // 仅重绑全局指针
            return;
        }
        std::swap(m_bufs.d_pos_curr, m_bufs.d_pos_next);
        if (m_bufs.usePosHalf) std::swap(m_bufs.d_pos_curr_h4, m_bufs.d_pos_next_h4);
        m_swappedThisFrame = true;
        UploadSimPosTableConst(m_bufs.d_pos_curr, m_bufs.d_pos_next);
        RebindPositionGlobals(m_bufs.d_pos_curr, m_bufs.d_pos_next,
            m_bufs.d_pos_curr_h4, m_bufs.d_pos_next_h4);
    }

    bool Simulator::bindTimelineFence(HANDLE sharedFenceHandle) {
        if (!sharedFenceHandle) return false;
        cudaExternalSemaphoreHandleDesc desc{}; desc.type = cudaExternalSemaphoreHandleTypeD3D12Fence; desc.handle.win32.handle = sharedFenceHandle; desc.flags = 0;
        if (cudaImportExternalSemaphore(&m_extTimelineSem, &desc) != cudaSuccess) { std::fprintf(stderr, "[Sim][Timeline] import external fence failed\n"); return false; }
        m_simFenceValue = 0; return true;
    }
    // ===== 播种 =====
    void Simulator::seedBoxLattice(uint32_t nx, uint32_t ny, uint32_t nz,
        float3 origin, float spacing) {
        uint64_t nreq64=uint64_t(nx)*ny*nz;
        uint32_t N=(nreq64>UINT32_MAX)?UINT32_MAX:uint32_t(nreq64);

        if(N>m_bufs.capacity){
            m_bufs.allocate(N);
            m_grid.allocateIndices(N);
            std::vector<uint32_t> h_idx(N);

            for(uint32_t i=0;i<N;++i) h_idx[i]=i;
                CUDA_CHECK(cudaMemcpy(m_grid.d_indices,h_idx.data(),sizeof(uint32_t)*N,cudaMemcpyHostToDevice));
                m_graphDirty=true;
            }
        m_params.numParticles=N;
        std::vector<float4> h_pos(N);
        uint32_t idx=0;

        for(uint32_t iz=0;iz<nz && idx<N;++iz)
        for(uint32_t iy=0;iy<ny && idx<N;++iy)
        for(uint32_t ix=0;ix<nx && idx<N;++ix){
            float x=origin.x+ix*spacing;
            float y=origin.y+iy*spacing;
            float z=origin.z+iz*spacing;

            x=fminf(fmaxf(x,m_params.grid.mins.x),m_params.grid.maxs.x);
            y=fminf(fmaxf(y,m_params.grid.mins.y),m_params.grid.maxs.y);
            z=fminf(fmaxf(z,m_params.grid.mins.z),m_params.grid.maxs.z);
            h_pos[idx++]=make_float4(x,y,z,1.0f);
        }
        for(;idx<N;++idx) h_pos[idx]=h_pos[N-1];
        const auto& c=console::Instance();

        if(c.sim.initial_jitter_enable){
            float h_use=(m_params.kernel.h>0.f)?m_params.kernel.h:spacing;
            float amp=c.sim.initial_jitter_scale_h*h_use;

            if(amp>0.f){
                std::mt19937 jrng(c.sim.initial_jitter_seed);
                std::uniform_real_distribution<float> J(-1.f,1.f);

                for(uint32_t i=0;i<N;++i){
                    float ox,oy,oz;

                    for(;;){
                        ox=J(jrng); oy=J(jrng); oz=J(jrng);
                        if(ox*ox+oy*oy+oz*oz<=1.f) break;
                    }

                    ox*=amp; oy*=amp; oz*=amp;
                    float4 p4=h_pos[i];
                    p4.x=fminf(fmaxf(p4.x+ox,m_params.grid.mins.x),m_params.grid.maxs.x);
                    p4.y=fminf(fmaxf(p4.y+oy,m_params.grid.mins.y),m_params.grid.maxs.y);
                    p4.z=fminf(fmaxf(p4.z+oz,m_params.grid.mins.z),m_params.grid.maxs.z);
                    h_pos[i]=p4;
                }
            }
        }
        CUDA_CHECK(cudaMemcpy(m_bufs.d_pos_curr,h_pos.data(),sizeof(float4)*N,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m_bufs.d_pos_next,h_pos.data(),sizeof(float4)*N,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(m_bufs.d_vel,0, sizeof(float4) * N));
        if (m_bufs.anyHalf()) m_bufs.packAllToHalf(N,0);
    }

 void Simulator::seedCubeMix(uint32_t groupCount,const float3* centers,uint32_t edgeParticles,float spacing,bool applyJitter,float jitterAmp,uint32_t jitterSeed){ if(groupCount==0||edgeParticles==0){ m_params.numParticles=0; return;} uint64_t per=uint64_t(edgeParticles)*edgeParticles*edgeParticles; uint64_t total64=per*uint64_t(groupCount); uint32_t total=(total64>UINT32_MAX)?UINT32_MAX:uint32_t(total64); if(total>m_bufs.capacity){ m_bufs.allocate(total); m_grid.allocateIndices(total); std::vector<uint32_t> h_idx(total); for(uint32_t i=0;i<total;++i) h_idx[i]=i; CUDA_CHECK(cudaMemcpy(m_grid.d_indices,h_idx.data(),sizeof(uint32_t)*total,cudaMemcpyHostToDevice)); m_graphDirty=true; } m_params.numParticles=total; std::vector<float4> h_pos(total); std::vector<float4> h_vel(total,make_float4(0,0,0,0)); uint32_t cursor=0; for(uint32_t g=0; g<groupCount && cursor<total; ++g){ float3 c=centers[g]; float start=-0.5f*spacing*float(edgeParticles-1); for(uint32_t iz=0; iz<edgeParticles && cursor<total; ++iz) for(uint32_t iy=0; iy<edgeParticles && cursor<total; ++iy) for(uint32_t ix=0; ix<edgeParticles && cursor<total; ++ix){ float3 p3=make_float3(c.x+start+ix*spacing,c.y+start+iy*spacing,c.z+start+iz*spacing); p3.x=fminf(fmaxf(p3.x,m_params.grid.mins.x),m_params.grid.maxs.x); p3.y=fminf(fmaxf(p3.y,m_params.grid.mins.y),m_params.grid.maxs.y); p3.z=fminf(fmaxf(p3.z,m_params.grid.mins.z),m_params.grid.maxs.z); h_pos[cursor++]=make_float4(p3.x,p3.y,p3.z,(float)g); } } if(applyJitter && jitterAmp>0.f){ std::mt19937 jrng(jitterSeed); std::uniform_real_distribution<float> U(-1.f,1.f); for(uint32_t i=0;i<total;++i){ float ox,oy,oz; for(;;){ ox=U(jrng); oy=U(jrng); oz=U(jrng); if(ox*ox+oy*oy+oz*oz<=1.f) break;} ox*=jitterAmp; oy*=jitterAmp; oz*=jitterAmp; float4 p4=h_pos[i]; p4.x=fminf(fmaxf(p4.x+ox,m_params.grid.mins.x),m_params.grid.maxs.x); p4.y=fminf(fmaxf(p4.y+oy,m_params.grid.mins.y),m_params.grid.maxs.y); p4.z=fminf(fmaxf(p4.z+oz,m_params.grid.mins.z),m_params.grid.maxs.z); h_pos[i]=p4; } } CUDA_CHECK(cudaMemcpy(m_bufs.d_pos_curr,h_pos.data(),sizeof(float4)*total,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(m_bufs.d_pos_next,h_pos.data(),sizeof(float4)*total,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(m_bufs.d_vel,h_vel.data(),sizeof(float4)*total,cudaMemcpyHostToDevice)); if(m_bufs.anyHalf()) m_bufs.packAllToHalf(total,0); }

 bool Simulator::computeStats(SimStats& out, uint32_t sampleStride) const { out={}; if(m_params.numParticles==0||m_bufs.capacity==0) return true; if(m_useHashedGrid){ LaunchCellRanges(m_grid.d_cellStart,m_grid.d_cellEnd,m_grid.d_cellKeys_sorted,m_params.numParticles,m_numCells,m_stream); CUDA_CHECK(cudaStreamSynchronize(m_stream)); } double avgN=0.0,avgV=0.0,avgRhoRel_g=0.0,avgR_g=0.0; uint32_t stride=(sampleStride==0?1u:sampleStride); bool ok=LaunchComputeStats(m_bufs.d_pos_next,m_bufs.d_vel,m_grid.d_indices_sorted,m_grid.d_cellStart,m_grid.d_cellEnd,m_params.grid,m_params.kernel,m_params.particleMass,m_params.numParticles,m_numCells,stride,&avgN,&avgV,&avgRhoRel_g,&avgR_g,m_stream); if(!ok) return false; out.N=m_params.numParticles; out.avgNeighbors=avgN; out.avgSpeed=avgV; out.avgRho=avgR_g; out.avgRhoRel=(m_params.restDensity>0.f)?(avgR_g/(double)m_params.restDensity):0.0; return true; }
 bool Simulator::computeStatsBruteforce(SimStats& out, uint32_t sampleStride, uint32_t maxISamples) const { if(m_params.numParticles==0){ out={}; return true; } double avgN=0.0,avgV=0.0,avgRhoRel=0.0,avgR=0.0; bool ok=LaunchComputeStatsBruteforce(m_bufs.d_pos_next,m_bufs.d_vel,m_params.kernel,m_params.particleMass,m_params.numParticles,(sampleStride==0?1u:sampleStride),maxISamples,&avgN,&avgV,&avgRhoRel,&avgR,m_stream); if(!ok) return false; out.N=m_params.numParticles; out.avgNeighbors=avgN; out.avgSpeed=avgV; out.avgRho=avgR; out.avgRhoRel=(m_params.restDensity>0.f)?(avgR/(double)m_params.restDensity):0.0; return true; }
}