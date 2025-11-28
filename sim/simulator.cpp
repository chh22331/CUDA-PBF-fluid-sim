#include "simulator.h"
#include "math_utils.h"
#include "poisson_disk.h"
#include "numeric_utils.h"
#include "logging.h"
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
#include "device_globals.cuh"
#include "simulation_context.h"
#include <limits>
#include "../engine/gfx/renderer.h"
#include "grid_strategy_dense.h"
#include "device_pos_state.cuh"
#include "grid_strategy_hashed.h"
#include "graph_builder.h"
#include "phase_pipeline.h"
#include "post_ops.h"

extern "C" void LaunchCellRanges(uint32_t* d_cellStart,
    uint32_t* d_cellEnd,
    const uint32_t* d_cellKeys_sorted,
    uint32_t N,
    uint32_t numCells,
    cudaStream_t stream);

#ifndef CUDA_CHECK
#define CUDA_CHECK(expr) \
    do { \
        cudaError_t _err = (expr); \
        if (_err != cudaSuccess) { \
            sim::Log(sim::LogChannel::Error, "CUDA %s (%d)", cudaGetErrorString(_err), (int)_err); \
        } \
    } while (0)
#endif

// ===================== D2D Memcpy 调试插桩补丁 =====================
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
    // ===== Utility =====
    static inline bool hostPtrReadable(const void* p) {
#ifdef _WIN32
        if (!p) return false;
        MEMORY_BASIC_INFORMATION mbi{};
        if (VirtualQuery(p, &mbi, sizeof(mbi)) == 0) return false;
        if (mbi.State != MEM_COMMIT) return false;
        if (mbi.Protect & PAGE_NOACCESS) return false;
        if (mbi.Protect & PAGE_GUARD) return false;
        return true;
#else
        return p != nullptr;
#endif
    }

} // namespace sim

#define CUDA_LOGGED_MEMCPY_D2D(TAG, DST, SRC, BYTES)                 \
    do {                                                             \
        sim::LogD2DMemcpy(TAG, (const void*)(DST), (const void*)(SRC), (size_t)(BYTES), false); \
        CUDA_CHECK(cudaMemcpy((DST), (SRC), (BYTES), cudaMemcpyDeviceToDevice));                \
    } while (0)

#define CUDA_LOGGED_MEMCPY_D2D_ASYNC(TAG, DST, SRC, BYTES, STREAM)   \
    do {                                                             \
        sim::LogD2DMemcpy(TAG, (const void*)(DST), (const void*)(SRC), (size_t)(BYTES), true);  \
        CUDA_CHECK(cudaMemcpyAsync((DST), (SRC), (BYTES), cudaMemcpyDeviceToDevice, (STREAM))); \
    } while (0)
// ===================== End D2D Memcpy 插桩 =====================

// ===== 外部 CUDA kernel =====
extern "C" void LaunchHashKeys(uint32_t*, uint32_t*, const float4*, sim::GridBounds, uint32_t, cudaStream_t);
extern "C" void LaunchCellRanges(uint32_t*, uint32_t*, const uint32_t*, uint32_t, uint32_t, cudaStream_t);
extern "C" void LaunchCellRangesCompact(uint32_t*, uint32_t*, uint32_t*, const uint32_t*, uint32_t, cudaStream_t);
extern "C" void LaunchLambda(float*, const float4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchLambdaCompact(float*, const float4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchDeltaApply(float4*, float4*, const float*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchDeltaApplyCompact(float4*, float4*, const float*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchXSPH(float4*, const float4*, const float4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchXSPHCompact(float4*, const float4*, const float4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" bool EnsureCellCompactScratch(uint32_t, uint32_t);
extern "C" void LaunchSortPairsQuery(size_t*, const uint32_t*, uint32_t*, const uint32_t*, uint32_t*, uint32_t, cudaStream_t);
extern "C" void LaunchSortPairs(void*, size_t, uint32_t*, uint32_t*, uint32_t*, uint32_t*, uint32_t, cudaStream_t);
extern "C" void LaunchVelocityGlobals(float dtInv, uint32_t N, cudaStream_t);
extern "C" void LaunchLambdaDenseGlobals(const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchLambdaCompactGlobals(const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchDeltaApplyDenseGlobals(const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchDeltaApplyCompactGlobals(const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchXsphDenseGlobals(const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchXsphCompactGlobals(const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchIntegratePred(float4*, const float4*, float4*, float3, float, uint32_t, cudaStream_t);
extern "C" void LaunchVelocity(float4*, const float4*, const float4*, float, uint32_t, cudaStream_t);
extern "C" void LaunchBoundary(float4*, float4*, sim::GridBounds, float, uint32_t, cudaStream_t);

namespace sim {
    uint64_t g_simFrameIndex = 0;

    void Simulator::performPingPongSwap(uint32_t N) {
        if (!m_ctx.pingPongPos) return;
        if (!m_bufs.externalPingPong) return;
        if (N == 0) return;

        float4* oldCurr = m_bufs.d_pos_curr;
        float4* oldNext = m_bufs.d_pos_next;
        m_bufs.swapPositionPingPong();
        m_swappedThisFrame = true;
        UploadSimPosTableConst(m_bufs.d_pos_curr, m_bufs.d_pos_next);

        if (m_graphExec) {
            patchGraphPositionPointers(oldCurr, oldNext);
        }
    }

    // Add helper to patch kernel node params with new position pointers
    void Simulator::patchGraphPositionPointers(float4* oldCurr, float4* oldNext) {
        cudaGraphExec_t exec = m_graphExec;
        if (!exec || m_posNodes.empty()) return;

        const int scanLimit = 512;
        int patchedPtr = 0, patchedGrid = 0;
        for (auto nd : m_posNodes) {
            cudaKernelNodeParams kp{};
            if (cudaGraphKernelNodeGetParams(nd, &kp) != cudaSuccess) continue;
            if (!kp.kernelParams) continue;

            uint32_t blocks = (m_params.numParticles + 255u) / 256u;
            if (blocks == 0) blocks = 1;
            if (kp.gridDim.x != blocks) { kp.gridDim.x = blocks; ++patchedGrid; }

            void** params = (void**)kp.kernelParams;
            bool modified = false;
            for (int i = 0; i < scanLimit; ++i) {
                void* slot = params[i];
                if (!slot) break;
                if (!hostPtrReadable(slot)) break;
                void* inner = *(void**)slot;
                if (inner == oldCurr) {
                    *(void**)slot = (void*)m_bufs.d_pos_curr;
                    modified = true;
                }
                else if (inner == oldNext) {
                    *(void**)slot = (void*)m_bufs.d_pos_next;
                    modified = true;
                }
            }
            if (modified || patchedGrid) {
                cudaGraphExecKernelNodeSetParams(exec, nd, &kp);
                if (modified) ++patchedPtr;
            }
        }
        if ((patchedPtr || patchedGrid) && console::Instance().debug.printHints) {
            std::fprintf(stderr,
                "[Graph][PosPatch] patchedPtr=%d patchedGrid=%d curr=%p next=%p\n",
                patchedPtr, patchedGrid,
                (void*)m_bufs.d_pos_curr, (void*)m_bufs.d_pos_next);
        }
    }
 
    bool sim::Simulator::initialize(const SimParams& p) {
        prof::Range rInit("Sim.Initialize", prof::Color(0x10, 0x90, 0xF0));

        m_params = p;
        CUDA_CHECK(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));

        const auto& c = console::Instance();
        m_frameTimingEveryN = c.perf.frame_timing_every_n;
        m_frameTimingEnabled = (m_frameTimingEveryN != 0);
        m_useHashedGrid = c.perf.use_hashed_grid;

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

        m_bufs.allocate(capacity);
        m_grid.allocateIndices(capacity);
        m_grid.ensureCompactCapacity(capacity);
        if (!buildGrid(m_params)) return false;

        if (capacity > 0) {
            std::vector<uint32_t> h_idx(capacity);
            for (uint32_t i = 0; i < capacity; ++i) h_idx[i] = i;
            CUDA_CHECK(cudaMemcpy(m_grid.d_indices, h_idx.data(),
                                  sizeof(uint32_t) * capacity, cudaMemcpyHostToDevice));
        }

        // 初始：curr -> next，pred 设为 next（alias）
        if (p.numParticles > 0) {
            CUDA_CHECK(cudaMemcpy(m_bufs.d_pos_next, m_bufs.d_pos,
                                  sizeof(float4) * p.numParticles,
                                  cudaMemcpyDeviceToDevice));
            m_bufs.d_pos_pred = m_bufs.d_pos_next;
        }
        UploadSimPosTableConst(m_bufs.d_pos_curr, m_bufs.d_pos_next);

        m_canPingPongPos = true;
        m_graphDirty = true;
        m_captured = {};
        m_cachedNodesReady = false;
        m_nodeRecycle = nullptr;
        m_lastFrameMs = -1.0f;
        m_evCursor = 0;
        m_lastParamUpdateFrame = -1;
        m_swappedThisFrame = false;

        if (m_pipeline.full().empty()) {
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
            "[Init][SchemeB] curr=%p next=%p pred(alias next)=%p N=%u\n",
            (void*)m_bufs.d_pos_curr, (void*)m_bufs.d_pos_next,
            (void*)m_bufs.d_pos_pred, p.numParticles);

        return true;
    }

    void sim::Simulator::shutdown() {
        if (m_extPosPred) {
            cudaDestroyExternalMemory(m_extPosPred);
            m_extPosPred = nullptr;
            m_bufs.detachExternalPosPred();
        }
        if (m_extVelocityMem) {
            cudaDestroyExternalMemory(m_extVelocityMem);
            m_extVelocityMem = nullptr;
            m_extVelocityPtr = nullptr;
            m_extVelocityBytes = 0;
            m_extVelocityStride = 0;
        }
        if (m_extRenderHalf) {
            cudaDestroyExternalMemory(m_extRenderHalf);
            m_extRenderHalf = nullptr;
        }
        m_grid.releaseAll();
        if (m_graphExec) { cudaGraphExecDestroy(m_graphExec);  m_graphExec = nullptr; }
        if (m_graph) { cudaGraphDestroy(m_graph);          m_graph = nullptr; }

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
        m_nodeRecycle = nullptr;
        m_kpRecycleBase = {};
        m_cachedNodesReady = false;

        m_velNodes.clear();
        m_velNodeParamsBase.clear();
        m_cachedVelNodes = false;

        m_posNodes.clear();
        m_posNodeParamsBase.clear();
        m_cachedPosNodes = false;

        size_t n = 0;
        CUDA_CHECK(cudaGraphGetNodes(m_graph, nullptr, &n));
        if (!n) {
            m_cachedNodesReady = true;
            return true;
        }
        std::vector<cudaGraphNode_t> nodes(n);
        CUDA_CHECK(cudaGraphGetNodes(m_graph, nodes.data(), &n));

        for (auto nd : nodes) {
            cudaGraphNodeType t;
            CUDA_CHECK(cudaGraphNodeGetType(nd, &t));
            if (t != cudaGraphNodeTypeKernel) continue;
            cudaKernelNodeParams kp{};
            CUDA_CHECK(cudaGraphKernelNodeGetParams(nd, &kp));

            if (!kp.kernelParams) continue;
            void** params = (void**)kp.kernelParams;

            std::unordered_set<const void*> watchVel{ (const void*)m_bufs.d_vel, (const void*)m_bufs.d_delta };
            std::unordered_set<const void*> watchPos;
             
            if (m_bufs.d_pos_curr) watchPos.insert((const void*)m_bufs.d_pos_curr);
            if (m_bufs.d_pos_next) watchPos.insert((const void*)m_bufs.d_pos_next);
            if (m_bufs.d_pos)      watchPos.insert((const void*)m_bufs.d_pos);
            if (m_bufs.d_pos_pred) watchPos.insert((const void*)m_bufs.d_pos_pred);

            bool hasVel = false, hasPos = false;
            for (int i = 0; i < 32; ++i) {
                void* slot = params[i];
                if (!slot) break;
                if (!hostPtrReadable(slot)) break;
                void* val = *(void**)slot;
                if (watchVel.count(val)) hasVel = true;
                if (watchPos.count(val)) hasPos = true;
                if (hasVel && hasPos) break;
            }
            if (hasVel) { m_velNodes.push_back(nd); m_velNodeParamsBase.push_back(kp); }
            if (hasPos) { m_posNodes.push_back(nd); m_posNodeParamsBase.push_back(kp); }
        }

        m_cachedVelNodes = true;
        m_cachedPosNodes = true;
        m_cachedNodesReady = true;
        return true;
    }

    // ===== Graph 变化检测 =====
    bool Simulator::structuralGraphChange(const SimParams& p) const {
        if (!m_graphExec) return true;
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
        if (!approxEq(p.dt, m_captured.dt)) return true;
        if (!approxEq3(p.gravity, m_captured.gravity)) return true;
        if (!approxEq(p.restDensity, m_captured.restDensity)) return true;
        if (!kernelEqualRelaxed(p.kernel, m_captured.kernel)) return true;
        return false;
    }
 
    bool Simulator::captureGraphIfNeeded(const SimParams& p) {
        if (!m_graphDirty) return true;
        if (!m_graphPointersChecked) {
            if (!m_bufs.d_pos || !m_bufs.d_vel || !m_bufs.d_pos_pred) {
                std::fprintf(stderr,
                    "[Graph][Error] Required buffers null before capture (pos=%p vel=%p pos_pred=%p)\n",
                    (void*)m_bufs.d_pos, (void*)m_bufs.d_vel, (void*)m_bufs.d_pos_pred);
            }
            m_graphPointersChecked = true;
        }
        prof::Range r("Sim.GraphCapture", prof::Color(0xE0, 0x60, 0x20));
        GraphBuilder builder;
        auto result = builder.BuildStructural(*this, p);
        return result.structuralRebuilt && result.reuseSucceeded;
    }

    bool Simulator::updateGraphsParams(const SimParams& p) {
        if (!m_paramDirty) return true;
        if (!m_graphExec) return false;

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

        uint32_t capacity = m_bufs.capacity;
        if (m_params.maxParticles == 0) m_params.maxParticles = capacity;
        m_params.maxParticles = std::min<uint32_t>(m_params.maxParticles, capacity);

        if (m_params.numParticles > m_bufs.capacity) {
            m_bufs.allocate(m_params.numParticles);
            m_grid.allocateIndices(m_bufs.capacity);
            std::vector<uint32_t> h_idx(m_bufs.capacity);
            for (uint32_t i = 0; i < m_bufs.capacity; ++i) h_idx[i] = i;
            CUDA_CHECK(cudaMemcpy(m_grid.d_indices, h_idx.data(),
                sizeof(uint32_t) * m_bufs.capacity,
                cudaMemcpyHostToDevice));
            m_graphDirty = true;
            UploadSimPosTableConst(m_bufs.d_pos_curr, m_bufs.d_pos_next);
        }

        updateGridIfNeeded(m_params);

        bool structuralChanged = m_paramTracker.structuralChanged(m_params, m_numCells);
        bool dynamicChanged = (!structuralChanged) && m_paramTracker.dynamicChanged(m_params);
        if (structuralChanged) m_graphDirty = true;
        else if (dynamicChanged) m_paramDirty = true;

        if (m_graphDirty)      captureGraphIfNeeded(m_params);
        else if (m_paramDirty) updateGraphsParams(m_params);

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

        if (!m_graphExec) {
            if (console::Instance().debug.printErrors)
                std::fprintf(stderr, "[Step][Error] Graph not ready.\n");
            return false;
        }
 
        CUDA_CHECK(cudaGraphLaunch(m_graphExec, m_stream));

        performPingPongSwap(m_params.numParticles);
        publishExternalVelocity(m_params.numParticles);
        signalSimFence();

        ++m_frameIndex;
        return true;
    }

    // ===== 外部共享缓冲 =====
    bool Simulator::importPosPredFromD3D12(void* sharedHandleWin32, size_t bytes) {
        if (!sharedHandleWin32 || bytes == 0) return false;
        if (m_extPosPred) {
            cudaDestroyExternalMemory(m_extPosPred);
            m_extPosPred = nullptr;
            m_bufs.detachExternalPosPred();
        }
        cudaExternalMemoryHandleDesc memDesc{};
        memDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
        memDesc.handle.win32.handle = sharedHandleWin32;
        memDesc.size  = bytes;
        memDesc.flags = cudaExternalMemoryDedicated;
        CUDA_CHECK(cudaImportExternalMemory(&m_extPosPred, &memDesc));

        cudaExternalMemoryBufferDesc bufDesc{};
        bufDesc.offset = 0;
        bufDesc.size   = bytes;
        void* devPtr = nullptr;
        CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&devPtr, m_extPosPred, &bufDesc));
        m_bufs.bindExternalPosPred(reinterpret_cast<float4*>(devPtr));
        m_canPingPongPos = true;
        return true;
    }

    bool Simulator::bindExternalPosPingPong(void* sharedHandleA, size_t bytesA,
        void* sharedHandleB, size_t bytesB) {
        if (!sharedHandleA || !sharedHandleB || bytesA == 0 || bytesB == 0) {
            std::fprintf(stderr, "[ExternalPosPP][Error] invalid handles or sizes\n");
            return false;
        }
        // 确保已创建 CUDA 上下文（防止首次 D3D12 共享映射后立即 D2D 拷贝出现 invalid value）
        cudaFree(0);

        cudaExternalMemory_t extA = nullptr, extB = nullptr;
        cudaExternalMemoryHandleDesc descA{};
        descA.type = cudaExternalMemoryHandleTypeD3D12Resource;
        descA.handle.win32.handle = sharedHandleA;
        descA.size = bytesA;
        descA.flags = cudaExternalMemoryDedicated;

        cudaExternalMemoryHandleDesc descB{};
        descB.type = cudaExternalMemoryHandleTypeD3D12Resource;
        descB.handle.win32.handle = sharedHandleB;
        descB.size = bytesB;
        descB.flags = cudaExternalMemoryDedicated;

        if (cudaImportExternalMemory(&extA, &descA) != cudaSuccess) {
            std::fprintf(stderr, "[ExternalPosPP][Error] import A failed\n");
            return false;
        }
        if (cudaImportExternalMemory(&extB, &descB) != cudaSuccess) {
            std::fprintf(stderr, "[ExternalPosPP][Error] import B failed\n");
            cudaDestroyExternalMemory(extA);
            return false;
        }

        cudaExternalMemoryBufferDesc bufA{}; bufA.offset = 0; bufA.size = bytesA;
        cudaExternalMemoryBufferDesc bufB{}; bufB.offset = 0; bufB.size = bytesB;
        void* devPtrA = nullptr; void* devPtrB = nullptr;
        if (cudaExternalMemoryGetMappedBuffer(&devPtrA, extA, &bufA) != cudaSuccess) {
            std::fprintf(stderr, "[ExternalPosPP][Error] map A failed\n");
            cudaDestroyExternalMemory(extA);
            cudaDestroyExternalMemory(extB);
            return false;
        }
        if (cudaExternalMemoryGetMappedBuffer(&devPtrB, extB, &bufB) != cudaSuccess) {
            std::fprintf(stderr, "[ExternalPosPP][Error] map B failed\n");
            cudaDestroyExternalMemory(extA);
            cudaDestroyExternalMemory(extB);
            return false;
        }

        uint32_t capA = static_cast<uint32_t>(bytesA / sizeof(float4));
        uint32_t capB = static_cast<uint32_t>(bytesB / sizeof(float4));
        uint32_t cap = (capA < capB) ? capA : capB;
        if (cap == 0) {
            std::fprintf(stderr, "[ExternalPosPP][Error] capacity zero\n");
            cudaDestroyExternalMemory(extA);
            cudaDestroyExternalMemory(extB);
            return false;
        }

        if (m_extPosPred) { cudaDestroyExternalMemory(m_extPosPred); m_extPosPred = nullptr; }
        m_extPosPred = extA;
        m_extraExternalMemB = extB;

        m_bufs.bindExternalPosPingPong(reinterpret_cast<float4*>(devPtrA),
                                       reinterpret_cast<float4*>(devPtrB), cap);
        m_bufs.d_pos_pred = m_bufs.d_pos_next;
        m_canPingPongPos = true;
        m_ctx.pingPongPos = true;

        std::fprintf(stderr,
            "[ExternalPosPP][Ready][SchemeB] curr=%p next=%p pred(alias)=%p cap=%u\n",
            (void*)m_bufs.d_pos_curr, (void*)m_bufs.d_pos_next,
            (void*)m_bufs.d_pos_pred, cap);

        return true;
    }

    bool Simulator::bindExternalVelocityBuffer(void* sharedHandleWin32, size_t bytes, uint32_t strideBytes) {
        if (!sharedHandleWin32 || bytes == 0 || strideBytes == 0) {
            std::fprintf(stderr, "[ExternalVel][Error] invalid handle/bytes/stride\n");
            return false;
        }
        if (m_extVelocityMem) {
            cudaDestroyExternalMemory(m_extVelocityMem);
            m_extVelocityMem = nullptr;
            m_extVelocityPtr = nullptr;
            m_extVelocityBytes = 0;
            m_extVelocityStride = 0;
        }

        cudaExternalMemoryHandleDesc desc{}; desc.type = cudaExternalMemoryHandleTypeD3D12Resource;
        desc.handle.win32.handle = sharedHandleWin32;
        desc.size = bytes;
        desc.flags = cudaExternalMemoryDedicated;
        if (cudaImportExternalMemory(&m_extVelocityMem, &desc) != cudaSuccess) {
            std::fprintf(stderr, "[ExternalVel][Error] import failed\n");
            m_extVelocityMem = nullptr;
            return false;
        }

        cudaExternalMemoryBufferDesc buf{}; buf.offset = 0; buf.size = bytes;
        void* devPtr = nullptr;
        if (cudaExternalMemoryGetMappedBuffer(&devPtr, m_extVelocityMem, &buf) != cudaSuccess) {
            std::fprintf(stderr, "[ExternalVel][Error] map failed\n");
            cudaDestroyExternalMemory(m_extVelocityMem);
            m_extVelocityMem = nullptr;
            return false;
        }

        m_extVelocityPtr = devPtr;
        m_extVelocityBytes = bytes;
        m_extVelocityStride = strideBytes;
        return true;
    }

    // ===== 播种 =====
    void Simulator::seedBoxLattice(uint32_t nx, uint32_t ny, uint32_t nz,
                                   float3 origin, float spacing) {
        uint64_t nreq64 = uint64_t(nx) * ny * nz;
        uint32_t N = (nreq64 > UINT32_MAX) ? UINT32_MAX : uint32_t(nreq64);

        if (N > m_bufs.capacity) {
            m_bufs.allocate(N);
            m_grid.allocateIndices(N);
            std::vector<uint32_t> h_idx(N);
            for (uint32_t i = 0; i < N; ++i) h_idx[i] = i;
            CUDA_CHECK(cudaMemcpy(m_grid.d_indices, h_idx.data(),
                                  sizeof(uint32_t) * N, cudaMemcpyHostToDevice));
            m_graphDirty = true;
        }

        m_params.numParticles = N;
        std::vector<float4> h_pos(N);
        uint32_t idx = 0;
        for (uint32_t iz = 0; iz < nz && idx < N; ++iz)
            for (uint32_t iy = 0; iy < ny && idx < N; ++iy)
                for (uint32_t ix = 0; ix < nx && idx < N; ++ix) {
                    float x = origin.x + ix * spacing;
                    float y = origin.y + iy * spacing;
                    float z = origin.z + iz * spacing;
                    x = fminf(fmaxf(x, m_params.grid.mins.x), m_params.grid.maxs.x);
                    y = fminf(fmaxf(y, m_params.grid.mins.y), m_params.grid.maxs.y);
                    z = fminf(fmaxf(z, m_params.grid.mins.z), m_params.grid.maxs.z);
                    h_pos[idx++] = make_float4(x, y, z, 1.0f);
                }
        for (; idx < N; ++idx) h_pos[idx] = h_pos[N - 1];

        const auto& c = console::Instance();
        if (c.sim.initial_jitter_enable) {
            float h_use = (m_params.kernel.h > 0.f) ? m_params.kernel.h : spacing;
            float amp = c.sim.initial_jitter_scale_h * h_use;
            if (amp > 0.f) {
                std::mt19937 jrng(c.sim.initial_jitter_seed);
                std::uniform_real_distribution<float> J(-1.f, 1.f);
                for (uint32_t i = 0; i < N; ++i) {
                    float ox, oy, oz;
                    for (;;) {
                        ox = J(jrng); oy = J(jrng); oz = J(jrng);
                        if (ox * ox + oy * oy + oz * oz <= 1.f) break;
                    }
                    ox *= amp; oy *= amp; oz *= amp;
                    float4 p4 = h_pos[i];
                    p4.x = fminf(fmaxf(p4.x + ox, m_params.grid.mins.x), m_params.grid.maxs.x);
                    p4.y = fminf(fmaxf(p4.y + oy, m_params.grid.mins.y), m_params.grid.maxs.y);
                    p4.z = fminf(fmaxf(p4.z + oz, m_params.grid.mins.z), m_params.grid.maxs.z);
                    h_pos[i] = p4;
                }
            }
        }

        CUDA_CHECK(cudaMemcpy(m_bufs.d_pos, h_pos.data(), sizeof(float4) * N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m_bufs.d_pos_next, h_pos.data(), sizeof(float4) * N, cudaMemcpyHostToDevice));
        m_bufs.d_pos_pred = m_bufs.d_pos_next;
        CUDA_CHECK(cudaMemset(m_bufs.d_vel, 0, sizeof(float4) * N));
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

    bool Simulator::computeCenterOfMass(float3& outCom, uint32_t sampleStride) const {
        outCom = make_float3(0, 0, 0);
        uint32_t N = m_params.numParticles;
        if (N == 0) return true;
        std::vector<float4> h_pos(N);
        CUDA_CHECK(cudaMemcpy(h_pos.data(), m_bufs.d_pos_next,
            sizeof(float4) * N, cudaMemcpyDeviceToHost));
        uint64_t cnt = 0;
        double sx = 0, sy = 0, sz = 0;
        uint32_t stride = (sampleStride == 0 ? 1u : sampleStride);
        for (uint32_t i = 0; i < N; i += stride) {
            sx += h_pos[i].x; sy += h_pos[i].y; sz += h_pos[i].z; ++cnt;
        }
        if (cnt == 0) return true;
        double inv = 1.0 / double(cnt);
        outCom = make_float3(float(sx * inv), float(sy * inv), float(sz * inv));
        return true;
    }

    void Simulator::seedCubeMix(uint32_t groupCount, const float3* centers, uint32_t edgeParticles,
        float spacing, bool applyJitter, float jitterAmp, uint32_t jitterSeed) {
        if (groupCount == 0 || edgeParticles == 0) {
            m_params.numParticles = 0;
            return;
        }
        uint64_t per = (uint64_t)edgeParticles * edgeParticles * edgeParticles;
        uint64_t total64 = per * (uint64_t)groupCount;
        uint32_t total = (total64 > UINT32_MAX) ? UINT32_MAX : (uint32_t)total64;

        if (total > m_bufs.capacity) {
            m_bufs.allocate(total);
            m_grid.allocateIndices(total);
            std::vector<uint32_t> h_idx(total);
            for (uint32_t i = 0; i < total; ++i) h_idx[i] = i;
            CUDA_CHECK(cudaMemcpy(m_grid.d_indices, h_idx.data(),
                                  sizeof(uint32_t) * total, cudaMemcpyHostToDevice));
            m_graphDirty = true;
        }

        m_params.numParticles = total;
        std::vector<float4> h_pos(total);
        std::vector<float4> h_vel(total, make_float4(0, 0, 0, 0));

        uint32_t cursor = 0;
        for (uint32_t g = 0; g < groupCount && cursor < total; ++g) {
            float3 c = centers[g];
            float start = -0.5f * spacing * float(edgeParticles - 1);
            for (uint32_t iz = 0; iz < edgeParticles && cursor < total; ++iz)
                for (uint32_t iy = 0; iy < edgeParticles && cursor < total; ++iy)
                    for (uint32_t ix = 0; ix < edgeParticles && cursor < total; ++ix) {
                        float3 p3 = make_float3(c.x + start + ix * spacing,
                                                c.y + start + iy * spacing,
                                                c.z + start + iz * spacing);
                        p3.x = fminf(fmaxf(p3.x, m_params.grid.mins.x), m_params.grid.maxs.x);
                        p3.y = fminf(fmaxf(p3.y, m_params.grid.mins.y), m_params.grid.maxs.y);
                        p3.z = fminf(fmaxf(p3.z, m_params.grid.mins.z), m_params.grid.maxs.z);
                        h_pos[cursor++] = make_float4(p3.x, p3.y, p3.z, (float)g);
                    }
        }

        if (applyJitter && jitterAmp > 0.f) {
            std::mt19937 jrng(jitterSeed);
            std::uniform_real_distribution<float> U(-1.f, 1.f);
            for (uint32_t i = 0; i < total; ++i) {
                float ox, oy, oz;
                for (;;) {
                    ox = U(jrng); oy = U(jrng); oz = U(jrng);
                    if (ox * ox + oy * oy + oz * oz <= 1.f) break;
                }
                ox *= jitterAmp; oy *= jitterAmp; oz *= jitterAmp;
                float4 p4 = h_pos[i];
                p4.x = fminf(fmaxf(p4.x + ox, m_params.grid.mins.x), m_params.grid.maxs.x);
                p4.y = fminf(fmaxf(p4.y + oy, m_params.grid.mins.y), m_params.grid.maxs.y);
                p4.z = fminf(fmaxf(p4.z + oz, m_params.grid.mins.z), m_params.grid.maxs.z);
                h_pos[i] = p4;
            }
        }

        CUDA_CHECK(cudaMemcpy(m_bufs.d_pos, h_pos.data(), sizeof(float4) * total, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m_bufs.d_pos_next, h_pos.data(), sizeof(float4) * total, cudaMemcpyHostToDevice));
        m_bufs.d_pos_pred = m_bufs.d_pos_next;
        CUDA_CHECK(cudaMemcpy(m_bufs.d_vel, h_vel.data(), sizeof(float4) * total, cudaMemcpyHostToDevice));
    }

    void Simulator::syncForRender() {
        if (m_stream) {
            cudaStreamSynchronize(m_stream);
        }
    }

    bool Simulator::bindTimelineFence(HANDLE sharedFenceHandle){
        if(!sharedFenceHandle) return false;
        cudaExternalSemaphoreHandleDesc desc{};
        desc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
        desc.handle.win32.handle = sharedFenceHandle;
        desc.flags = 0;
        if(cudaImportExternalSemaphore(&m_extTimelineSem,&desc) != cudaSuccess){
            std::fprintf(stderr,"[Sim][Timeline] import external fence failed\n");
            return false;
        }
        m_simFenceValue = 0;
        return true;
    }

    void Simulator::publishExternalVelocity(uint32_t count) {
        if (!m_extVelocityPtr || count == 0) return;
        if (m_extVelocityStride == 0) return;
        size_t bytes = size_t(count) * size_t(m_extVelocityStride);
        if (bytes > m_extVelocityBytes) bytes = m_extVelocityBytes;
        if (bytes == 0) return;
        CUDA_CHECK(cudaMemcpyAsync(m_extVelocityPtr, m_bufs.d_vel, bytes, cudaMemcpyDeviceToDevice, m_stream));
    }

    void Simulator::signalSimFence(){
        if(!m_extTimelineSem) return;
        ++m_simFenceValue;
        cudaExternalSemaphoreSignalParams params{};
        params.params.fence.value = m_simFenceValue;
        params.flags = 0;
        cudaSignalExternalSemaphoresAsync(&m_extTimelineSem,&params,1,m_stream);
    }

    // ===== Graph 速度指针热更新 =====
    void Simulator::patchGraphVelocityPointers(const float4* fromPtr, const float4* toPtr) {
        if (!fromPtr || !toPtr || fromPtr == toPtr) return;
        cudaGraphExec_t exec = m_graphExec;
        if (!exec) return;
        auto& nodes = m_velNodes;
        if (nodes.empty()) return;
        const int scanLimit = 256;
        int patched = 0;
        for (auto nd : nodes) {
            cudaKernelNodeParams kp{};
            if (cudaGraphKernelNodeGetParams(nd, &kp) != cudaSuccess) continue;
            if (!kp.kernelParams) continue;
            void** params = (void**)kp.kernelParams;
            bool modified = false;
            for (int i = 0; i < scanLimit; ++i) {
                void* slot = params[i];
                if (!slot) break;
                if (!hostPtrReadable(slot)) break;
                void* inner = *(void**)slot;
                if (inner == (const void*)fromPtr) {
                    *(void**)slot = (void*)toPtr;
                    modified = true;
                }
            }
            if (modified) {
                if (cudaGraphExecKernelNodeSetParams(exec, nd, &kp) == cudaSuccess)
                    ++patched;
            }
        }
        if (patched && console::Instance().debug.printHints) {
            std::fprintf(stderr,
                "[Graph][VelPatch] from=%p to=%p patched=%d\n",
                (const void*)fromPtr, (const void*)toPtr, patched);
        }
    }
} // namespace sim
