#include "simulator.h"
#include "math_utils.h"
#include "poisson_disk.h"
#include "numeric_utils.h"
#include "emit_params.h"
#include "stats.h"              // extern kernels
#include "logging.h"
#include "emitter.h"
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <unordered_set>
#include "grid_system.h"
#include "../engine/core/console.h"
#include "../engine/core/prof_nvtx.h" // add NVTX
#include "precision_traits.cuh"
#include "device_globals.cuh"
#include "precision_stage.h"
#include "simulation_context.h"
#include <limits>
#include "grid_strategy_dense.h"
#include "grid_strategy_hashed.h"
#include "graph_builder.h"      // NEW: graph building refactor
#include "phase_pipeline.h"      // NEW: phase pipeline
#include "post_ops.h"            // NEW: post ops

#ifndef CUDA_CHECK
#define CUDA_CHECK(expr)                                                            \
    do {                                                                            \
        cudaError_t _err = (expr);                                                  \
        if (_err != cudaSuccess) {                                                  \
            sim::Log(sim::LogChannel::Error, "CUDA %s (%d)",                       \
                     cudaGetErrorString(_err), (int)_err);                          \
        }                                                                           \
    } while (0)
#endif

// ===== External kernel launches (order preserved) =====
extern "C" void LaunchIntegratePred(float4*, const float4*, float4*, float3, float, uint32_t, cudaStream_t);
extern "C" void LaunchIntegratePredMP(const float4*, const float4*, float4*, const sim::Half4*, const sim::Half4*, float3, float, uint32_t, cudaStream_t);
extern "C" void LaunchHashKeys(uint32_t*, uint32_t*, const float4*, sim::GridBounds, uint32_t, cudaStream_t);
extern "C" void LaunchHashKeysMP(uint32_t*, uint32_t*, const float4*, const sim::Half4*, sim::GridBounds, uint32_t, cudaStream_t);
extern "C" void LaunchCellRanges(uint32_t*, uint32_t*, const uint32_t*, uint32_t, uint32_t, cudaStream_t);
extern "C" void LaunchCellRangesCompact(uint32_t*, uint32_t*, uint32_t*, const uint32_t*, uint32_t, cudaStream_t);
extern "C" void LaunchLambda(float*, const float4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchLambdaCompact(float*, const float4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchLambdaCompactMP(float*, const float4*, const sim::Half4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchDeltaApply(float4*, float4*, const float*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchDeltaApplyCompact(float4*, float4*, const float*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchDeltaApplyCompactMP(float4*, float4*, const float*, const float4*, const sim::Half4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchLambdaMP(float*, const float4*, const sim::Half4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchDeltaApplyMP(float4*, float4*, const float*, const float4*, const sim::Half4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchXSPH(float4*, const float4*, const float4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchXSPHCompact(float4*, const float4*, const float4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchXSPHCompactMP(float4*, const float4*, const sim::Half4*, const float4*, const sim::Half4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" void LaunchXSPHMP(float4*, const float4*, const sim::Half4*, const float4*, const sim::Half4*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, sim::DeviceParams, uint32_t, cudaStream_t);
extern "C" bool EnsureCellCompactScratch(uint32_t, uint32_t);
extern "C" void LaunchVelocity(float4*, const float4*, const float4*, float, uint32_t, cudaStream_t);
extern "C" void LaunchVelocityMP(float4*, const float4*, const float4*, const sim::Half4*, const sim::Half4*, float, uint32_t, cudaStream_t);
extern "C" void LaunchBoundary(float4*, float4*, sim::GridBounds, float, uint32_t, cudaStream_t);
extern "C" void LaunchSortPairsQuery(size_t*, const uint32_t*, uint32_t*, const uint32_t*, uint32_t*, uint32_t, cudaStream_t);
extern "C" void LaunchSortPairs(void*, size_t, uint32_t*, uint32_t*, uint32_t*, uint32_t*, uint32_t, cudaStream_t);
extern "C" void LaunchRecycleToNozzleConst(float4*, float4*, float4*, sim::GridBounds, float, uint32_t, int, cudaStream_t);
extern "C" void* GetRecycleKernelPtr();

namespace sim {
// 全局帧索引定义
uint64_t g_simFrameIndex = 0;

// 前向声明以便在下方使用（定义仍在后面）
static inline bool hostPtrReadable(const void* p);

// Add helper to patch kernel node params with new position pointers
void Simulator::patchGraphPositionPointers(bool fullGraph, float4* oldCurr, float4* oldNext) {
    const auto& c = console::Instance();
    if (!c.perf.graph_hot_update_enable) return;
    cudaGraphExec_t exec = fullGraph ? m_graphExecFull : m_graphExecCheap;
    if (!exec) return;

    // 只使用缓存的包含 position 指针的节点列表（cacheGraphNodes 已构建）
    auto& nodes = fullGraph ? m_posNodesFull : m_posNodesCheap;
    if (nodes.empty()) return;

    int scanLimit = (c.perf.graph_hot_update_scan_limit > 0 &&
        c.perf.graph_hot_update_scan_limit < 256)
        ? c.perf.graph_hot_update_scan_limit : 64;

    int patched = 0;
    for (auto nd : nodes) {
        cudaKernelNodeParams kp{};
        if (cudaGraphKernelNodeGetParams(nd, &kp) != cudaSuccess) continue;
        if (!kp.kernelParams) continue;

        void** params = (void**)kp.kernelParams;
        bool modified = false;

        // 扫描有限槽位（只在首次遇到旧指针时替换）
        for (int i = 0; i < scanLimit; ++i) {
            void* slot = params[i];
            if (!slot) break;

            // 直接指针匹配
            if (slot == (void*)oldCurr) {
                params[i] = (void*)m_bufs.d_pos_curr;
                modified = true;
                continue;
            }
            if (slot == (void*)oldNext) {
                params[i] = (void*)m_bufs.d_pos_next;
                modified = true;
                continue;
            }

            // 主机二级存储（参数块指针）匹配
            if (!hostPtrReadable(slot)) continue;
            void* inner = *(void**)slot;
            if (inner == (void*)oldCurr) {
                *(void**)slot = (void*)m_bufs.d_pos_curr;
                modified = true;
            }
            else if (inner == (void*)oldNext) {
                *(void**)slot = (void*)m_bufs.d_pos_next;
                modified = true;
            }
        }

        if (modified) {
            cudaError_t err = cudaGraphExecKernelNodeSetParams(exec, nd, &kp);
            if (err == cudaSuccess) {
                ++patched;
            }
            else {
                std::fprintf(stderr,
                    "[Graph][PatchPos][Warn] setParams failed err=%d (%s)\n",
                    (int)err, cudaGetErrorString(err));
            }
        }
    }

    if (patched > 0) {
        m_graphNodesPatchedOnce = true;
        if (c.debug.printHints) {
            std::fprintf(stderr,
                "[Graph][HotUpdate] Patched position pointers on %d kernel nodes (full=%d)\n",
                patched, fullGraph ? 1 : 0);
        }
    }
}

// ===== Utility =====
static inline bool hostPtrReadable(const void* p) {
#ifdef _WIN32
    MEMORY_BASIC_INFORMATION mbi{};
    if (!p) return false;
    if (VirtualQuery(p, &mbi, sizeof(mbi)) == 0) return false;
    if (mbi.State != MEM_COMMIT) return false;
    if (mbi.Protect & PAGE_NOACCESS) return false;
    if (mbi.Protect & PAGE_GUARD) return false;
    return true;
#else
    return p != nullptr;
#endif
}

// ===== Initialization / Shutdown =====
bool Simulator::initialize(const SimParams& p) {
    prof::Range rInit("Sim.Initialize", prof::Color(0x10, 0x90, 0xF0));

    m_params = p;
    CUDA_CHECK(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));

    const auto& c = console::Instance();
    m_frameTimingEveryN = c.perf.frame_timing_every_n;
    m_frameTimingEnabled = (m_frameTimingEveryN != 0);
    m_useHashedGrid = c.perf.use_hashed_grid;

    // 启用位置 ping-pong（除非后续绑定外部预测缓冲）
    m_canPingPongPos = true;

    // 策略上下文
    m_ctx.bufs = &m_bufs;
    m_ctx.grid = &m_grid;
    m_ctx.useHashedGrid = m_useHashedGrid;
    if (m_useHashedGrid) {
        m_gridStrategy = std::make_unique<HashedGridStrategy>();
    }
    else {
        m_gridStrategy = std::make_unique<DenseGridStrategy>();
    }
    m_ctx.gridStrategy = m_gridStrategy.get();
    m_ctx.dispatcher = &m_kernelDispatcher;

    uint32_t capacity = (p.maxParticles > 0) ? p.maxParticles : p.numParticles;
    if (capacity == 0) capacity = 1;
    bool needHalf = (p.precision.positionStore == NumericType::FP16_Packed) ||
        (p.precision.velocityStore == NumericType::FP16_Packed) ||
        (p.precision.predictedPosStore == NumericType::FP16_Packed);
    if (needHalf) m_bufs.allocateWithPrecision(p.precision, capacity); else m_bufs.allocate(capacity);
    UpdateDevicePrecisionView(m_bufs, p.precision);

    if (!buildGrid(p)) {
        std::fprintf(stderr, "[Init][Error] buildGrid failed (invalid grid dims)\n");
        return false;
    }
    m_grid.allocateIndices(capacity);
    m_grid.ensureCompactCapacity(capacity);
    {
        std::vector<uint32_t> h_idx(capacity);
        for (uint32_t i = 0; i < capacity; ++i) h_idx[i] = i;
        CUDA_CHECK(cudaMemcpy(m_grid.d_indices, h_idx.data(), sizeof(uint32_t) * capacity, cudaMemcpyHostToDevice));
    }

    if (p.numParticles > 0 && m_bufs.d_pos_curr && m_bufs.d_pos_next) {
        prof::Range rCopy("Init.D2D.pos_curr->pos_next", prof::Color(0xE0, 0x30, 0x30));
        CUDA_CHECK(cudaMemcpy(m_bufs.d_pos_next, m_bufs.d_pos_curr, sizeof(float4) * p.numParticles, cudaMemcpyDeviceToDevice));
        if (needHalf) m_bufs.packAllToHalf(p.numParticles, m_stream);
    }

    m_graphDirty = true;
    std::fprintf(stderr,
        "[Debug] posPredExternal=%d pos_curr=%p pos_next=%p extPosHandle=%p\n",
        (int)m_bufs.posPredExternal,
        (void*)m_bufs.d_pos_curr,
        (void*)m_bufs.d_pos_next,
        (void*)m_extPosPred);
    return true;
}

void Simulator::shutdown() {
    if (m_extPosPred) {
        cudaDestroyExternalMemory(m_extPosPred);
        m_extPosPred = nullptr;
        m_bufs.detachExternalPosPred();
    }

    m_grid.releaseAll();

    if (m_graphExecFull)  { cudaGraphExecDestroy(m_graphExecFull);  m_graphExecFull  = nullptr; }
    if (m_graphFull)      { cudaGraphDestroy(m_graphFull);          m_graphFull      = nullptr; }
    if (m_graphExecCheap) { cudaGraphExecDestroy(m_graphExecCheap); m_graphExecCheap = nullptr; }
    if (m_graphCheap)     { cudaGraphDestroy(m_graphCheap);         m_graphCheap     = nullptr; }

    for (int i = 0; i < 2; ++i) {
        if (m_evStart[i]) { cudaEventDestroy(m_evStart[i]); m_evStart[i] = nullptr; }
        if (m_evEnd[i])   { cudaEventDestroy(m_evEnd[i]);   m_evEnd[i]   = nullptr; }
    }

    if (m_stream) { cudaStreamDestroy(m_stream); m_stream = nullptr; }
}

// ===== Grid & Param Helpers =====
bool Simulator::buildGrid(const SimParams& p) {
    int3 dim   = GridSystem::ComputeDims(p.grid);
    m_numCells = GridSystem::NumCells(dim);
    if (m_numCells == 0) return false;
    m_grid.allocGridRanges(m_numCells); // fix typo
    m_params.grid = p.grid;
    m_params.grid.dim = dim;
    return true;
}

bool Simulator::updateGridIfNeeded(const SimParams& p) {
    int3 dim       = GridSystem::ComputeDims(p.grid);
    uint32_t newNC = GridSystem::NumCells(dim);
    bool changed   = false;

    if (newNC != m_numCells) changed = true;
    if (!gridEqual(p.grid, m_captured.grid)) changed = true;

    if (changed) {
        m_grid.resizeGridRanges(newNC);
        m_numCells   = newNC;
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

// ===== Graph Node Caching (Recycle/Velocity updates) =====
bool Simulator::cacheGraphNodes() {
    m_nodeRecycleFull  = nullptr;
    m_nodeRecycleCheap = nullptr;
    m_kpRecycleBaseFull = {};
    m_kpRecycleBaseCheap = {};
    m_cachedNodesReady = false;

    m_velNodesFull.clear();
    m_velNodesCheap.clear();
    m_velNodeParamsBaseFull.clear();
    m_velNodeParamsBaseCheap.clear();
    m_cachedVelNodes = false;

    void* target = GetRecycleKernelPtr();

    auto scan = [&](cudaGraph_t g,
                    std::vector<cudaGraphNode_t>& outVel, std::vector<cudaKernelNodeParams>& outVelParams,
                    std::vector<cudaGraphNode_t>& outPos, std::vector<cudaKernelNodeParams>& outPosParams,
                    bool recordFull) {
        if (!g) return;
        size_t n = 0;
        CUDA_CHECK(cudaGraphGetNodes(g, nullptr, &n));
        if (!n) return;
        std::vector<cudaGraphNode_t> nodes(n);
        CUDA_CHECK(cudaGraphGetNodes(g, nodes.data(), &n));

        for (auto nd : nodes) {
            cudaGraphNodeType t;
            CUDA_CHECK(cudaGraphNodeGetType(nd, &t));
            if (t != cudaGraphNodeTypeKernel) continue;
            cudaKernelNodeParams kp{};
            CUDA_CHECK(cudaGraphKernelNodeGetParams(nd, &kp));

            if (recordFull && kp.func == target) {
                m_nodeRecycleFull  = nd;
                m_kpRecycleBaseFull = kp;
            } else if (!recordFull && kp.func == target) {
                m_nodeRecycleCheap  = nd;
                m_kpRecycleBaseCheap = kp;
            }

            if (!kp.kernelParams) continue;
            void** params = (void**)kp.kernelParams;
            std::unordered_set<const void*> watchVel{ (const void*)m_bufs.d_vel, (const void*)m_bufs.d_delta };
            std::unordered_set<const void*> watchPos{ (const void*)m_bufs.d_pos, (const void*)m_bufs.d_pos_pred };
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
            if (hasVel) { outVel.push_back(nd); outVelParams.push_back(kp); }
            if (hasPos) { outPos.push_back(nd); outPosParams.push_back(kp); }
        }
    };

    scan(m_graphFull,  m_velNodesFull,  m_velNodeParamsBaseFull,  m_posNodesFull,  m_posNodeParamsBaseFull,  true);
    scan(m_graphCheap, m_velNodesCheap, m_velNodeParamsBaseCheap, m_posNodesCheap, m_posNodeParamsBaseCheap, false);

    m_cachedVelNodes  = true;
    m_cachedPosNodes  = true;
    m_cachedNodesReady = true;
    return true;
}

// ===== Legacy Structural / Param Change Detection (kept for now) =====
bool Simulator::structuralGraphChange(const SimParams& p) const {
    if (!m_graphExecFull || !m_graphExecCheap) return true;
    if (p.solverIters   != m_captured.solverIters)   return true;
    if (p.maxNeighbors  != m_captured.maxNeighbors)  return true;
    if (p.numParticles  != m_captured.numParticles)  return true;

    int3 dim    = GridSystem::ComputeDims(p.grid);
    uint32_t nc = GridSystem::NumCells(dim);
    if (nc != m_captured.numCells) return true;
    if (!gridEqual(p.grid, m_captured.grid)) return true;
    return false;
}

bool Simulator::paramOnlyGraphChange(const SimParams& p) const {
    if (structuralGraphChange(p)) return false; // structural takes precedence
    float dtRel = fabsf(p.dt - m_captured.dt) /
                  fmaxf(1e-9f, fmaxf(p.dt, m_captured.dt));
    if (dtRel < 0.002f &&
        approxEq3(p.gravity, m_captured.gravity) &&
        approxEq(p.restDensity, m_captured.restDensity) &&
        kernelEqualRelaxed(p.kernel, m_captured.kernel)) {
        return false;
    }
    if (!approxEq(p.dt,          m_captured.dt))          return true;
    if (!approxEq3(p.gravity,    m_captured.gravity))     return true;
    if (!approxEq(p.restDensity, m_captured.restDensity)) return true;
    if (!kernelEqualRelaxed(p.kernel, m_captured.kernel)) return true;
    return false;
}

// ===== Graph Build (delegated to GraphBuilder) =====
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

    prof::Range r("Sim.GraphCapture", prof::Color(0xE0,0x60,0x20));
    GraphBuilder builder;
    auto result = builder.BuildStructural(*this, p);
    return result.structuralRebuilt && result.reuseSucceeded;
}

bool Simulator::updateGraphsParams(const SimParams& p) {
    if (!m_paramDirty) return true;
    if (!m_graphExecFull || !m_graphExecCheap) return false;

    const auto& c = console::Instance();

    prof::Range r("Sim.GraphUpdate", prof::Color(0xA0,0x50,0x10));
    GraphBuilder builder;
    auto res = builder.UpdateDynamic(*this, p,
                                     c.perf.graph_param_update_min_interval,
                                     m_frameIndex,
                                     m_lastParamUpdateFrame);

    if (m_graphDirty) { // dynamic update failed → structural rebuild fallback
        auto rs = builder.BuildStructural(*this, p);
        if (!rs.structuralRebuilt) return false;
        m_lastParamUpdateFrame = m_frameIndex;
        return true;
    }

    if (res.dynamicUpdated) {
        m_lastParamUpdateFrame = m_frameIndex;
        return true;
    }
    return !m_paramDirty; // false only if still dirty
}

// ===== Phase Kernels (kept as-is for now) =====
void Simulator::kHashKeys(cudaStream_t s, const SimParams& p) {
    prof::Range r("Phase.HashKeys", prof::Color(0x60,0xB0,0x40));
    bool useMP = UseHalfForPosition(p, Stage::GridBuild, m_bufs);
    if (useMP)
        LaunchHashKeysMP(m_grid.d_cellKeys, m_grid.d_indices, m_bufs.d_pos_next, m_bufs.d_pos_pred_h4, p.grid, p.numParticles, s);
    else
        LaunchHashKeys(m_grid.d_cellKeys, m_grid.d_indices, m_bufs.d_pos_next, p.grid, p.numParticles, s);
}

void Simulator::kSort(cudaStream_t s, const SimParams& p) {
    prof::Range r("Phase.SortPairs", prof::Color(0x90,0x50,0xF0));
    LaunchSortPairs(m_grid.d_sortTemp, m_grid.sortTempBytes,
                    m_grid.d_cellKeys, m_grid.d_cellKeys_sorted,
                    m_grid.d_indices,  m_grid.d_indices_sorted,
                    p.numParticles, s);
}

void Simulator::kCellRanges(cudaStream_t s, const SimParams& p) {
    prof::Range r("Phase.CellRanges.Dense", prof::Color(0x40,0xD0,0xD0));
    CUDA_CHECK(cudaMemsetAsync(m_grid.d_cellStart, 0xFF, sizeof(uint32_t) * m_numCells, s));
    CUDA_CHECK(cudaMemsetAsync(m_grid.d_cellEnd,   0xFF, sizeof(uint32_t) * m_numCells, s));
    LaunchCellRanges(m_grid.d_cellStart, m_grid.d_cellEnd, m_grid.d_cellKeys_sorted,
                     p.numParticles, m_numCells, s);
}

void Simulator::kCellRangesCompact(cudaStream_t s, const SimParams& p) {
    prof::Range r("Phase.CellRanges.Dense", prof::Color(0x40,0xD0,0xD0));
    LaunchCellRangesCompact(m_grid.d_cellUniqueKeys, m_grid.d_cellOffsets, m_grid.d_compactCount,
                            m_grid.d_cellKeys_sorted, p.numParticles, s);
    m_numCompactCells = 0; // placeholder until used
}

void Simulator::kSolveIter(cudaStream_t s, const SimParams& p) {
    prof::Range r("Phase.SolveIter", prof::Color(0xE0,0x80,0x40));
    DeviceParams dp = MakeDeviceParams(p);
    // Use predicted/next buffer as working positions (same semantic as old d_pos_pred)
    if (m_useHashedGrid) {
        bool useMP = UseHalfForPosition(p, Stage::LambdaSolve, m_bufs);
        if (useMP) {
            LaunchLambdaCompactMP(m_bufs.d_lambda, m_bufs.d_pos_next, m_bufs.d_pos_pred_h4,
                                  m_grid.d_indices_sorted, m_grid.d_cellKeys_sorted,
                                  m_grid.d_cellUniqueKeys, m_grid.d_cellOffsets, m_grid.d_compactCount,
                                  dp, p.numParticles, s);
            LaunchDeltaApplyCompactMP(m_bufs.d_pos_next, m_bufs.d_delta, m_bufs.d_lambda,
                                      m_bufs.d_pos_next, m_bufs.d_pos_pred_h4,
                                      m_grid.d_indices_sorted, m_grid.d_cellKeys_sorted,
                                      m_grid.d_cellUniqueKeys, m_grid.d_cellOffsets, m_grid.d_compactCount,
                                      dp, p.numParticles, s);
        } else {
            LaunchLambdaCompact(m_bufs.d_lambda, m_bufs.d_pos_next,
                                 m_grid.d_indices_sorted, m_grid.d_cellKeys_sorted,
                                 m_grid.d_cellUniqueKeys, m_grid.d_cellOffsets, m_grid.d_compactCount,
                                 dp, p.numParticles, s);
            LaunchDeltaApplyCompact(m_bufs.d_pos_next, m_bufs.d_delta, m_bufs.d_lambda,
                                    m_grid.d_indices_sorted, m_grid.d_cellKeys_sorted,
                                    m_grid.d_cellUniqueKeys, m_grid.d_cellOffsets, m_grid.d_compactCount,
                                    dp, p.numParticles, s);
        }
    } else {
        bool useMP = UseHalfForPosition(p, Stage::LambdaSolve, m_bufs);
        if (useMP) {
            LaunchLambdaMP(m_bufs.d_lambda, m_bufs.d_pos_next, m_bufs.d_pos_pred_h4,
                           m_grid.d_indices_sorted, m_grid.d_cellKeys_sorted,
                           m_grid.d_cellStart, m_grid.d_cellEnd,
                           dp, p.numParticles, s);
            LaunchDeltaApplyMP(m_bufs.d_pos_next, m_bufs.d_delta, m_bufs.d_lambda,
                               m_bufs.d_pos_next, m_bufs.d_pos_pred_h4,
                               m_grid.d_indices_sorted, m_grid.d_cellKeys_sorted,
                               m_grid.d_cellStart, m_grid.d_cellEnd,
                               dp, p.numParticles, s);
        } else {
            LaunchLambda(m_bufs.d_lambda, m_bufs.d_pos_next,
                         m_grid.d_indices_sorted, m_grid.d_cellKeys_sorted,
                         m_grid.d_cellStart, m_grid.d_cellEnd,
                         dp, p.numParticles, s);
            LaunchDeltaApply(m_bufs.d_pos_next, m_bufs.d_delta, m_bufs.d_lambda,
                              m_grid.d_indices_sorted, m_grid.d_cellKeys_sorted,
                              m_grid.d_cellStart, m_grid.d_cellEnd,
                              dp, p.numParticles, s);
        }
    }
    LaunchBoundary(m_bufs.d_pos_next, m_bufs.d_vel_curr, p.grid, 0.0f, p.numParticles, s);
}

void Simulator::kVelocityAndPost(cudaStream_t s, const SimParams& p) {
    prof::Range r("Phase.VelocityPost", prof::Color(0xC0,0x40,0xA0));
    bool useMP = UseHalfForPosition(p, Stage::VelocityUpdate, m_bufs);
    if (useMP)
        LaunchVelocityMP(m_bufs.d_vel_curr, m_bufs.d_pos_curr, m_bufs.d_pos_next,
                         m_bufs.d_pos_h4, m_bufs.d_pos_pred_h4,
                         1.0f / p.dt, p.numParticles, s);
    else
        LaunchVelocity(m_bufs.d_vel_curr, m_bufs.d_pos_curr, m_bufs.d_pos_next,
                       1.0f / p.dt, p.numParticles, s);

    // XSPH now writes directly into current velocity if enabled (handled in post ops in new pipeline path)
    float4* effectiveVel = m_bufs.d_vel_curr;
    LaunchBoundary(m_bufs.d_pos_next, effectiveVel, p.grid, p.boundaryRestitution, p.numParticles, s);
    LaunchRecycleToNozzleConst(m_bufs.d_pos_curr, m_bufs.d_pos_next, effectiveVel,
                               p.grid, p.dt, p.numParticles, 0, s);
    // Removed device-to-device copies; swap will occur after graph / pipeline launch.
}

void Simulator::kIntegratePred(cudaStream_t s, const SimParams& p) {
    prof::Range r("Phase.Integrate", prof::Color(0x50,0xA0,0xFF));
    bool useMP = (UseHalfForPosition(p, Stage::Integration, m_bufs) && UseHalfForVelocity(p, Stage::Integration, m_bufs));
    if (useMP)
        LaunchIntegratePredMP(m_bufs.d_pos_curr, m_bufs.d_vel_curr, m_bufs.d_pos_next,
                              m_bufs.d_pos_h4, m_bufs.d_vel_h4,
                              p.gravity, p.dt, p.numParticles, s);
    else
        LaunchIntegratePred(m_bufs.d_pos_curr, m_bufs.d_vel_curr, m_bufs.d_pos_next,
                            p.gravity, p.dt, p.numParticles, s);
    LaunchBoundary(m_bufs.d_pos_next, m_bufs.d_vel_curr, p.grid, 0.0f, p.numParticles, s);
}

// ===== Step =====
bool Simulator::step(const SimParams& p) {
    prof::Range rf("Sim.Step", prof::Color(0x30, 0x30, 0xA0));
    g_simFrameIndex = m_frameIndex;
    // 帧开始守卫与状态快照
    static bool     s_prevPosPredExternal = false;
    static bool     s_prevPingPong = false;
    bool curExt = m_bufs.posPredExternal;
    bool curPP = m_ctx.pingPongPos;

    if (curExt != s_prevPosPredExternal) {
        std::fprintf(stderr,
            "[StateChange][Frame=%llu] posPredExternal %d -> %d\n",
            (unsigned long long)m_frameIndex,
            (int)s_prevPosPredExternal, (int)curExt);
    }
    if (curPP != s_prevPingPong) {
        std::fprintf(stderr,
            "[StateChange][Frame=%llu] pingPongPos %d -> %d (pre-eval)\n",
            (unsigned long long)m_frameIndex,
            (int)s_prevPingPong, (int)curPP);
    }

    s_prevPosPredExternal = curExt;
    s_prevPingPong = curPP;
    m_params = p;
    bool allowPP = console::Instance().perf.allow_pingpong_with_external_pred;
    bool expected = m_canPingPongPos && (!m_bufs.posPredExternal || allowPP);
    if (m_ctx.pingPongPos != expected) {
        std::fprintf(stderr,
            "[PingPong][Adjust][Frame=%llu] overwrite pingPongPos (source=Simulator::step) cur=%d expected=%d ext=%d allow=%d can=%d\n",
            (unsigned long long)m_frameIndex,
            (int)m_ctx.pingPongPos, (int)expected,
            (int)m_bufs.posPredExternal, (int)allowPP, (int)m_canPingPongPos);
        m_ctx.pingPongPos = expected;
    }
    const auto& cHot = console::Instance();
    if (m_frameTimingEveryN != cHot.perf.frame_timing_every_n) {
        m_frameTimingEveryN = cHot.perf.frame_timing_every_n;
        m_frameTimingEnabled = (m_frameTimingEveryN != 0);
    }
    {
        prof::Range rEmit("EmitParticles", prof::Color(0xFF, 0x90, 0x30));
        m_params.grid.dim = GridSystem::ComputeDims(m_params.grid);
    }
    const auto& c = console::Instance();
    EmitParams ep{}; console::BuildEmitParams(c, ep);
    {
        uint32_t capacity = m_bufs.capacity;
        if (m_params.maxParticles == 0) m_params.maxParticles = capacity;
        m_params.maxParticles = std::min<uint32_t>(m_params.maxParticles, capacity);
        uint32_t emitted = Emitter::EmitFaucet(m_bufs, m_params, c, ep, m_frameIndex, m_stream);
        if (emitted > 0) m_bufs.packAllToHalf(m_params.numParticles, m_stream);
    }
    if (m_params.numParticles > m_bufs.capacity) {
        const auto& pr = m_params.precision;
        bool needHalf = (pr.positionStore == NumericType::FP16_Packed || pr.positionStore == NumericType::FP16 ||
            pr.velocityStore == NumericType::FP16_Packed || pr.velocityStore == NumericType::FP16 ||
            pr.predictedPosStore == NumericType::FP16_Packed || pr.predictedPosStore == NumericType::FP16);
        if (needHalf) m_bufs.allocateWithPrecision(pr, m_params.numParticles); else m_bufs.allocate(m_params.numParticles);
        UpdateDevicePrecisionView(m_bufs, m_params.precision);
        m_grid.allocateIndices(m_bufs.capacity);
        std::vector<uint32_t> h_idx(m_bufs.capacity); for (uint32_t i = 0; i < m_bufs.capacity; ++i) h_idx[i] = i;
        {
            prof::Range rIdx("Expand.D2D.initIndices", prof::Color(0xD0, 0x50, 0x50));
            CUDA_CHECK(cudaMemcpy(m_grid.d_indices, h_idx.data(), sizeof(uint32_t) * m_bufs.capacity, cudaMemcpyHostToDevice));
        }
        if (needHalf) std::fprintf(stderr, "[Precision] Reallocate newCap=%u pos_h4=%p vel_h4=%p pred_h4=%p\n",
            m_bufs.capacity, (void*)m_bufs.d_pos_h4, (void*)m_bufs.d_vel_h4, (void*)m_bufs.d_pos_pred_h4);
        m_graphDirty = true;
    }
    SetEmitParamsAsync(&ep, m_stream);
    updateGridIfNeeded(m_params);
    bool structuralChanged = m_paramTracker.structuralChanged(m_params, m_numCells);
    bool dynamicChanged = (!structuralChanged) && m_paramTracker.dynamicChanged(m_params);
    if (structuralChanged) m_graphDirty = true; else if (dynamicChanged) m_paramDirty = true;
    if (m_graphDirty) captureGraphIfNeeded(m_params); else if (m_paramDirty) updateGraphsParams(m_params);
 
    int everyN = (c.perf.sort_compact_every_n <= 0) ? 1 : c.perf.sort_compact_every_n;
    bool needFull = (m_frameIndex == 0) || (m_lastFullFrame < 0) ||
        ((m_frameIndex - m_lastFullFrame) >= everyN) ||
        (m_params.numParticles != m_captured.numParticles);
    prof::Mark(needFull ? "Launch.FullGraph" : "Launch.CheapGraph",
        needFull ? prof::Color(0xD0, 0x40, 0x20) : prof::Color(0x20, 0xA0, 0x40));
    bool doTiming = m_frameTimingEnabled && (m_frameTimingEveryN <= 1 || (m_frameIndex % m_frameTimingEveryN) == 0);
    int cur = (m_evCursor & 1); int prev = ((m_evCursor + 1) & 1);
    if (doTiming) {
        for (int i = 0; i < 2; ++i) {
            if (!m_evStart[i]) cudaEventCreate(&m_evStart[i]);
            if (!m_evEnd[i])   cudaEventCreate(&m_evEnd[i]);
        }
        if (m_evStart[cur] && m_evEnd[cur]) CUDA_CHECK(cudaEventRecord(m_evStart[cur], m_stream));
    }

    auto updNode = [&](cudaGraphExec_t exec, cudaGraphNode_t& node, cudaKernelNodeParams& base) {
        if (!exec || !node || m_params.numParticles == 0) return;
        uint32_t blocks = (m_params.numParticles + 255u) / 256u;
        if (blocks == 0) blocks = 1;
        cudaKernelNodeParams kp = base;
        kp.gridDim = dim3(blocks, 1, 1);
        cudaError_t err = cudaGraphExecKernelNodeSetParams(exec, node, &kp);
        if (err == cudaErrorInvalidResourceHandle) {
            std::fprintf(stderr, "[Graph][Warn] Kernel node invalid, recache & retry (node=%p, N=%u)\n", (void*)node, m_params.numParticles);
            cacheGraphNodes();
            if (exec == m_graphExecFull) { node = m_nodeRecycleFull; base = m_kpRecycleBaseFull; }
            else if (exec == m_graphExecCheap) { node = m_nodeRecycleCheap; base = m_kpRecycleBaseCheap; }
            if (node) {
                kp = base; kp.gridDim = dim3(blocks, 1, 1);
                cudaError_t err2 = cudaGraphExecKernelNodeSetParams(exec, node, &kp);
                if (err2 != cudaSuccess)
                    std::fprintf(stderr, "[Graph][Error] Retry setParams failed err=%d (%s)\n", (int)err2, cudaGetErrorString(err2));
            }
            else {
                std::fprintf(stderr, "[Graph][Error] Recycle node not found after recache.\n");
            }
        }
        };

    bool useGraphs = c.perf.use_cuda_graphs;
    std::fprintf(stderr,
        "[PingPong][Frame=%llu][PreLaunch] useGraphs=%d needFull=%d pingPongPos=%d ext=%d allow=%d can=%d pos_curr=%p pos_next=%p pred=%p\n",
        (unsigned long long)m_frameIndex, (int)useGraphs, (int)needFull, (int)m_ctx.pingPongPos,
        (int)m_bufs.posPredExternal, (int)allowPP, (int)m_canPingPongPos,
        (void*)m_bufs.d_pos_curr, (void*)m_bufs.d_pos_next, (void*)m_bufs.d_pos_pred);

    if (useGraphs) {
        if (!m_graphExecFull || !m_graphExecCheap) {
            if (c.debug.printErrors)
                std::fprintf(stderr, "Simulator::step: CUDA Graph not ready (exec == nullptr)\n");
            return false;
        }
        if (needFull) updNode(m_graphExecFull, m_nodeRecycleFull, m_kpRecycleBaseFull);
        else          updNode(m_graphExecCheap, m_nodeRecycleCheap, m_kpRecycleBaseCheap);
        if (needFull) {
            CUDA_CHECK(cudaGraphLaunch(m_graphExecFull, m_stream));
            m_lastFullFrame = m_frameIndex;
        }
        else {
            CUDA_CHECK(cudaGraphLaunch(m_graphExecCheap, m_stream));
        }
    }
    else {
        if (m_pipeline.full().empty()) {
            BuildDefaultPipelines(m_pipeline);
            PostOpsConfig cfg{};
            cfg.enableXsph = (m_params.xsph_c > 0.f);
            cfg.enableBoundary = true;
            cfg.enableRecycle = true;
            m_pipeline.post().configure(cfg, m_useHashedGrid, cfg.enableXsph);
        }
        m_ctx.bufs = &m_bufs; m_ctx.grid = &m_grid; m_ctx.useHashedGrid = m_useHashedGrid;
        m_ctx.gridStrategy = m_gridStrategy.get(); m_ctx.dispatcher = &m_kernelDispatcher;
        if (needFull) {
            m_pipeline.runFull(m_ctx, m_params, m_stream);
            m_lastFullFrame = m_frameIndex;
        }
        else {
            m_pipeline.runCheap(m_ctx, m_params, m_stream);
        }
    }
 
    // 执行位置 ping-pong（仅在允许且未使用外部预测缓冲时）
    if (m_ctx.pingPongPos) {
        float4* oldCurr = m_bufs.d_pos_curr;
        float4* oldNext = m_bufs.d_pos_next;
        std::fprintf(stderr,
            "[PingPong][SwapAttempt][Frame=%llu] beforeSwap curr=%p next=%p pred=%p ext=%d allow=%d\n",
            (unsigned long long)m_frameIndex, (void*)oldCurr, (void*)oldNext, (void*)m_bufs.d_pos_pred,
            (int)m_bufs.posPredExternal, (int)allowPP);
        m_bufs.swapPositionPingPong();
        if (useGraphs && c.perf.graph_hot_update_enable) {
            patchGraphPositionPointers(true, oldCurr, oldNext);
            patchGraphPositionPointers(false, oldCurr, oldNext);
        }
        std::fprintf(stderr,
            "[PingPong][SwapDone][Frame=%llu] curr=%p next=%p pred=%p ext=%d\n",
            (unsigned long long)m_frameIndex, (void*)m_bufs.d_pos_curr, (void*)m_bufs.d_pos_next, (void*)m_bufs.d_pos_pred,
            (int)m_bufs.posPredExternal);
    } else {
        std::fprintf(stderr,
            "[PingPong][SkipSwap][Frame=%llu] pingPongPos=0 ext=%d allow=%d can=%d curr=%p next=%p pred=%p\n",
            (unsigned long long)m_frameIndex, (int)m_bufs.posPredExternal, (int)allowPP, (int)m_canPingPongPos,
            (void*)m_bufs.d_pos_curr, (void*)m_bufs.d_pos_next, (void*)m_bufs.d_pos_pred);
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
    std::fprintf(stderr, "[PingPongEval][Frame=%llu] posPredExternal=%d pingPongPos=%d curr=%p next=%p pred=%p\n",
    (unsigned long long)m_frameIndex, (int)m_bufs.posPredExternal, (int)m_ctx.pingPongPos,
    (void*)m_bufs.d_pos_curr, (void*)m_bufs.d_pos_next, (void*)m_bufs.d_pos_pred);

    // 修正：外部双缓冲 (externalPingPong) 允许 next==pred，不视为退化；仅内部/镜像单外部时才禁用
    if (m_ctx.pingPongPos) {
        if (!m_bufs.externalPingPong && (m_bufs.d_pos_next == m_bufs.d_pos_pred)) {
            std::fprintf(stderr,
                "[PingPong][Disable] d_pos_next == d_pos_pred (external binding collapsed both); auto-off. Frame=%llu\n",
                (unsigned long long)m_frameIndex);
            m_ctx.pingPongPos = false;
        }
    }
    // 在 step 末尾 PingPongEval 后，修正禁用逻辑与镜像同步（新增 mirror 频率）
    const auto& cfgPerf = console::Instance().perf;
    int mirrorEvery = (cfgPerf.graph_param_update_min_interval > 0 ? cfgPerf.graph_param_update_min_interval : 1); // 可换成新字段 perf.pos_mirror_every_n

    if (m_bufs.posPredExternal) {
        // 外部双缓冲模式下不做 Mirror（避免覆盖另一个工作缓冲导致粒子不移动）
        if (!m_bufs.externalPingPong) {
            // 仅在需要时镜像当前工作位置到外部（这里选择写 current，可改写 next）
            bool doMirror = (mirrorEvery == 1) || ((m_frameIndex % mirrorEvery) == 0);
            if (doMirror && m_params.numParticles > 0) {
                size_t bytes = sizeof(float4) * m_params.numParticles;
                cudaMemcpyAsync(m_bufs.d_pos_pred, m_bufs.d_pos_curr, bytes, cudaMemcpyDeviceToDevice, m_stream);
                std::fprintf(stderr,
                    "[ExternalPred][Mirror] Frame=%llu bytes=%zu curr=%p pred(external)=%p every=%d\n",
                    (unsigned long long)m_frameIndex, bytes, (void*)m_bufs.d_pos_curr, (void*)m_bufs.d_pos_pred, mirrorEvery);
            }
        }
        // 修正禁用条件：只在内部模式检查
    }
    else if (m_ctx.pingPongPos && (m_bufs.d_pos_next == m_bufs.d_pos_pred) && !m_bufs.externalPingPong) {
        std::fprintf(stderr,
            "[PingPong][Disable] d_pos_next == d_pos_pred (internal alias collapse); auto-off. Frame=%llu\n",
            (unsigned long long)m_frameIndex);
        m_ctx.pingPongPos = false;
    }

    ++m_frameIndex;
    return true;
}

bool Simulator::importPosPredFromD3D12(void* sharedHandleWin32, size_t bytes) {
    if (!sharedHandleWin32 || bytes == 0) return false;
    if (m_extPosPred) {
        cudaDestroyExternalMemory(m_extPosPred);
        m_extPosPred = nullptr;
        m_bufs.detachExternalPosPred();
    }

    cudaExternalMemoryHandleDesc memDesc{};
    memDesc.type                = cudaExternalMemoryHandleTypeD3D12Resource;
    memDesc.handle.win32.handle = sharedHandleWin32;
    memDesc.size                = bytes;
    memDesc.flags               = cudaExternalMemoryDedicated;
    CUDA_CHECK(cudaImportExternalMemory(&m_extPosPred, &memDesc));

    cudaExternalMemoryBufferDesc bufDesc{};
    bufDesc.offset = 0;
    bufDesc.size   = bytes;
    void* devPtr   = nullptr;
    CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&devPtr, m_extPosPred, &bufDesc));
    m_bufs.bindExternalPosPred(reinterpret_cast<float4*>(devPtr));
    m_canPingPongPos = true;
    return true;
}

bool Simulator::bindExternalPosPingPong(void* sharedHandleA, size_t bytesA, void* sharedHandleB, size_t bytesB) {
    if (!sharedHandleA || !sharedHandleB || bytesA == 0 || bytesB == 0) {
        std::fprintf(stderr, "[ExternalPosPP][Error] invalid handles or sizes A=%p B=%p bytesA=%zu bytesB=%zu\n", sharedHandleA, sharedHandleB, bytesA, bytesB);
        return false;
    }
    // 导入 A
    cudaExternalMemory_t extA = nullptr; cudaExternalMemory_t extB = nullptr;
    cudaExternalMemoryHandleDesc descA{}; descA.type = cudaExternalMemoryHandleTypeD3D12Resource; descA.handle.win32.handle = sharedHandleA; descA.size = bytesA; descA.flags = cudaExternalMemoryDedicated;
    cudaExternalMemoryHandleDesc descB{}; descB.type = cudaExternalMemoryHandleTypeD3D12Resource; descB.handle.win32.handle = sharedHandleB; descB.size = bytesB; descB.flags = cudaExternalMemoryDedicated;
    if (cudaImportExternalMemory(&extA, &descA) != cudaSuccess) { std::fprintf(stderr, "[ExternalPosPP][Error] import A failed\n"); return false; }
    if (cudaImportExternalMemory(&extB, &descB) != cudaSuccess) { std::fprintf(stderr, "[ExternalPosPP][Error] import B failed\n"); cudaDestroyExternalMemory(extA); return false; }
    cudaExternalMemoryBufferDesc bufA{}; bufA.offset = 0; bufA.size = bytesA; void* devPtrA = nullptr;
    cudaExternalMemoryBufferDesc bufB{}; bufB.offset = 0; bufB.size = bytesB; void* devPtrB = nullptr;
    if (cudaExternalMemoryGetMappedBuffer(&devPtrA, extA, &bufA) != cudaSuccess) { std::fprintf(stderr, "[ExternalPosPP][Error] map A failed\n"); cudaDestroyExternalMemory(extA); cudaDestroyExternalMemory(extB); return false; }
    if (cudaExternalMemoryGetMappedBuffer(&devPtrB, extB, &bufB) != cudaSuccess) { std::fprintf(stderr, "[ExternalPosPP][Error] map B failed\n"); cudaDestroyExternalMemory(extA); cudaDestroyExternalMemory(extB); return false; }

    uint32_t capA = static_cast<uint32_t>(bytesA / sizeof(float4));
    uint32_t capB = static_cast<uint32_t>(bytesB / sizeof(float4));
    uint32_t cap = (capA < capB) ? capA : capB;
    if (cap == 0) { std::fprintf(stderr, "[ExternalPosPP][Error] capacity zero after import\n"); cudaDestroyExternalMemory(extA); cudaDestroyExternalMemory(extB); return false; }

    // 保存外部句柄以便 shutdown 释放（复用已有 m_extPosPred 仅保存第一个，扩展：可新增数组，这里简化）
    if (m_extPosPred) { cudaDestroyExternalMemory(m_extPosPred); m_extPosPred = nullptr; }
    m_extPosPred = extA; // 仅保存 A；B 留在局部引用，需额外管理（简化：不销毁 B，后续可扩展存 vector）

    m_bufs.bindExternalPosPingPong(reinterpret_cast<float4*>(devPtrA), reinterpret_cast<float4*>(devPtrB), cap);
    m_canPingPongPos = true; m_ctx.pingPongPos = true;
    std::fprintf(stderr, "[ExternalPosPP][Ready] curr=%p next=%p cap=%u\n", (void*)m_bufs.d_pos_curr, (void*)m_bufs.d_pos_next, cap);
    return true;
}

void Simulator::seedBoxLattice(uint32_t nx, uint32_t ny, uint32_t nz,
                               float3 origin, float spacing) {
    uint64_t nreq64 = uint64_t(nx) * ny * nz;
    uint32_t N = (nreq64 > UINT32_MAX) ? UINT32_MAX : uint32_t(nreq64);

    if (N > m_bufs.capacity) {
        m_bufs.allocate(N);
        m_grid.allocateIndices(N);
        std::vector<uint32_t> h_idx(N);
        for (uint32_t i = 0; i < N; ++i) h_idx[i] = i;
        CUDA_CHECK(cudaMemcpy(m_grid.d_indices, h_idx.data(), sizeof(uint32_t) * N, cudaMemcpyHostToDevice));
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
    for (; idx < N; ++idx) h_pos[idx] = (N > 0) ? h_pos[N - 1] : make_float4(origin.x, origin.y, origin.z, 1.0f);

    const auto& c = console::Instance();
    if (c.sim.initial_jitter_enable) {
        float h_use = (m_params.kernel.h > 0.f) ? m_params.kernel.h : spacing;
        float amp   = c.sim.initial_jitter_scale_h * h_use;
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

    CUDA_CHECK(cudaMemcpy(m_bufs.d_pos,      h_pos.data(), sizeof(float4) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m_bufs.d_pos_pred, h_pos.data(), sizeof(float4) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(m_bufs.d_vel, 0, sizeof(float4) * N));
    if (m_bufs.anyHalf()) m_bufs.packAllToHalf(N, 0);
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
    uint32_t Nreq   = (nreq64 > UINT32_MAX) ? UINT32_MAX : (uint32_t)nreq64;
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
    outCom = make_float3(0,0,0);
    uint32_t N = m_params.numParticles;
    if (N == 0) return true;
    std::vector<float4> h_pos(N);
    CUDA_CHECK(cudaMemcpy(h_pos.data(), m_bufs.d_pos_pred, sizeof(float4) * N, cudaMemcpyDeviceToHost));
    uint64_t cnt = 0; double sx = 0, sy = 0, sz = 0;
    uint32_t stride = (sampleStride == 0 ? 1u : sampleStride);
    for (uint32_t i = 0; i < N; i += stride) { sx += h_pos[i].x; sy += h_pos[i].y; sz += h_pos[i].z; ++cnt; }
    if (cnt == 0) return true;
    double inv = 1.0 / double(cnt);
    outCom = make_float3(float(sx * inv), float(sy * inv), float(sz * inv));
    return true;
}

void Simulator::seedCubeMix(uint32_t groupCount, const float3* centers, uint32_t edgeParticles,
                            float spacing, bool applyJitter, float jitterAmp, uint32_t jitterSeed) {
    if (groupCount == 0 || edgeParticles == 0) { m_params.numParticles = 0; return; }
    uint64_t per = (uint64_t)edgeParticles * edgeParticles * edgeParticles;
    uint64_t total64 = per * (uint64_t)groupCount;
    uint32_t total = (total64 > UINT32_MAX) ? UINT32_MAX : (uint32_t)total64;

    if (total > m_bufs.capacity) {
        m_bufs.allocate(total);
        m_grid.allocateIndices(total);
        std::vector<uint32_t> h_idx(total);
        for (uint32_t i = 0; i < total; ++i) h_idx[i] = i;
        CUDA_CHECK(cudaMemcpy(m_grid.d_indices, h_idx.data(), sizeof(uint32_t) * total, cudaMemcpyHostToDevice));
        m_graphDirty = true;
    }

    m_params.numParticles = total;
    std::vector<float4> h_pos(total);
    std::vector<float4> h_vel(total, make_float4(0,0,0,0));

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
            for (;;) { ox = U(jrng); oy = U(jrng); oz = U(jrng); if (ox*ox + oy*oy + oz*oz <= 1.f) break; }
            ox *= jitterAmp; oy *= jitterAmp; oz *= jitterAmp;
            float4 p4 = h_pos[i];
            p4.x = fminf(fmaxf(p4.x + ox, m_params.grid.mins.x), m_params.grid.maxs.x);
            p4.y = fminf(fmaxf(p4.y + oy, m_params.grid.mins.y), m_params.grid.maxs.y);
            p4.z = fminf(fmaxf(p4.z + oz, m_params.grid.mins.z), m_params.grid.maxs.z);
            h_pos[i] = p4;
        }
    }

    CUDA_CHECK(cudaMemcpy(m_bufs.d_pos,      h_pos.data(), sizeof(float4) * total, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m_bufs.d_pos_pred, h_pos.data(), sizeof(float4) * total, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m_bufs.d_vel,      h_vel.data(), sizeof(float4) * total, cudaMemcpyHostToDevice));
    if (m_bufs.anyHalf()) m_bufs.packAllToHalf(total, 0);
}

bool Simulator::computeStats(SimStats& out, uint32_t sampleStride) const {
    out = {};
    if (m_params.numParticles == 0 || m_bufs.capacity == 0) return true;
    const auto& c = console::Instance();
    if (m_useHashedGrid) {
        LaunchCellRanges(m_grid.d_cellStart, m_grid.d_cellEnd, m_grid.d_cellKeys_sorted,
                          m_params.numParticles, m_numCells, m_stream);
        CUDA_CHECK(cudaStreamSynchronize(m_stream));
    }
    double avgN=0.0, avgV=0.0, avgRhoRel_g=0.0, avgR_g=0.0;
    uint32_t stride = (sampleStride==0?1u:sampleStride);
    bool ok = LaunchComputeStats(m_bufs.d_pos_pred, m_bufs.d_vel,
                                 m_grid.d_indices_sorted, m_grid.d_cellStart, m_grid.d_cellEnd,
                                 m_params.grid, m_params.kernel, m_params.particleMass,
                                 m_params.numParticles, m_numCells, stride,
                                 &avgN,&avgV,&avgRhoRel_g,&avgR_g, m_stream); // fix arg name
    if(!ok) return false;
    out.N = m_params.numParticles;
    out.avgNeighbors = avgN;
    out.avgSpeed = avgV;
    out.avgRho = avgR_g;
    out.avgRhoRel = (m_params.restDensity>0.f)? (avgR_g/(double)m_params.restDensity):0.0;
    bool wantDiag = (c.debug.printDiagnostics||c.debug.printWarnings||c.debug.printHints);
    uint64_t diagEvery = (uint64_t)(c.debug.logEveryN<=0?1:c.debug.logEveryN);
    if(!wantDiag || (m_frameIndex%diagEvery)!=0ull) return true;
    double avgN_bf=0.0, avgV_bf=0.0, avgRhoRel_bf=0.0, avgR_bf=0.0; const uint32_t kMaxISamples=2048;
    bool ok_bf = LaunchComputeStatsBruteforce(m_bufs.d_pos_pred, m_bufs.d_vel, m_params.kernel, m_params.particleMass,
                                              m_params.numParticles, stride, kMaxISamples,
                                              &avgN_bf,&avgV_bf,&avgRhoRel_bf,&avgR_bf, m_stream);
    if(!ok_bf) return true; // ignore diag failure
    bool hasCap = (m_params.maxNeighbors>0); double capN = hasCap? double(m_params.maxNeighbors): std::numeric_limits<double>::infinity();
    bool nearCap = hasCap && (avgN>=0.9*capN);
    bool severeUnder = (avgN_bf>0.0) && (avgN < 0.8*avgN_bf);
    double rhoRel_bf = (m_params.restDensity>0.f)? (avgR_bf/(double)m_params.restDensity):0.0;
    static uint64_t s_last = UINT64_MAX; bool shouldDiag = (nearCap||severeUnder);
    if(shouldDiag && s_last!=m_frameIndex && wantDiag){ s_last=m_frameIndex; double h=double(m_params.kernel.h); double cell=double(m_params.grid.cellSize); double ratio=(h>0.0)?(cell/h):0.0; if(c.debug.printDiagnostics){ std::fprintf(stderr,
        "[Diag] Frame=%llu | N=%u | h=%.6g | cell=%.3f (cell/h=%.3f) | dim=(%d,%d,%d) numCells=%u\n",
        (unsigned long long)m_frameIndex, m_params.numParticles, h, cell, ratio,
        m_params.grid.dim.x, m_params.grid.dim.y, m_params.grid.dim.z, m_numCells);
        std::fprintf(stderr,
        "[Diag] Neighbors: grid=%.3f, brute=%.3f | RhoRel: grid=%.3f, brute=%.3f | maxNeighbors=%d%s\n",
        avgN, avgN_bf, out.avgRhoRel, rhoRel_bf, m_params.maxNeighbors, (nearCap?" (near cap)":"")); }
        if(c.debug.printHints){ if(ratio<0.9) std::fprintf(stderr,"[Hint] cellSize/h <0.9 建议 ~[1.0,1.5]h\n"); else if(ratio>2.0) std::fprintf(stderr,"[Hint] cellSize/h >2.0 建议 ~[1.0,1.5]h\n"); if(nearCap) std::fprintf(stderr,"[Hint] avgNeighbors 接近上限 %d\n", m_params.maxNeighbors); if(severeUnder) std::fprintf(stderr,"[Hint] 网格邻居显著偏低\n"); }
    }
    return true;
}

bool Simulator::computeStatsBruteforce(SimStats& out, uint32_t sampleStride, uint32_t maxISamples) const {
    if(m_params.numParticles==0){ out={}; return true; }
    double avgN=0.0, avgV=0.0, avgRhoRel=0.0, avgR=0.0;
    bool ok = LaunchComputeStatsBruteforce(m_bufs.d_pos_pred, m_bufs.d_vel, m_params.kernel, m_params.particleMass,
                                           m_params.numParticles, (sampleStride==0?1u:sampleStride), maxISamples,
                                           &avgN,&avgV,&avgRhoRel,&avgR, m_stream);
    if(!ok) return false;
    out.N = m_params.numParticles;
    out.avgNeighbors = avgN;
    out.avgSpeed = avgV;
    out.avgRho = avgR;
    out.avgRhoRel = (m_params.restDensity>0.f)? (avgR/(double)m_params.restDensity):0.0;
    return true;
}

void Simulator::syncForRender() {
    if (m_stream) {
        cudaStreamSynchronize(m_stream);
    }
}

} // namespace sim