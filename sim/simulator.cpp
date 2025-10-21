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
#include "../engine/gfx/renderer.h"   
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
extern "C" void LaunchIntegratePredSemiImplicit(const float4*, float4*, float4*, float3, float, uint32_t, cudaStream_t);

namespace sim {
    // 全局帧索引定义
    uint64_t g_simFrameIndex = 0;

// 前向声明以便在下方使用（定义仍在后面）
static inline bool hostPtrReadable(const void* p);

// Add helper to patch kernel node params with new position pointers
void Simulator::patchGraphPositionPointers(bool fullGraph,
    float4* oldCurr,
    float4* oldNext,
    float4* oldPred) {
    const auto& c = console::Instance();
    if (!c.perf.graph_hot_update_enable) return;
    cudaGraphExec_t exec = fullGraph ? m_graphExecFull : m_graphExecCheap;
    if (!exec) return;

    auto& nodes = fullGraph ? m_posNodesFull : m_posNodesCheap;
    if (nodes.empty()) return;

    int userLimit = c.perf.graph_hot_update_scan_limit;
    int scanLimit = (userLimit > 0 && userLimit <= 4096) ? userLimit : 512;

    int patchedPtr = 0, patchedGrid = 0;
    for (auto nd : nodes) {
        cudaKernelNodeParams kp{};
        if (cudaGraphKernelNodeGetParams(nd, &kp) != cudaSuccess) continue;
        if (!kp.kernelParams) continue;

        uint32_t blocks = (m_params.numParticles + 255u) / 256u;
        if (blocks == 0) blocks = 1;
        bool gridChanged = false;
        if (kp.gridDim.x != blocks) {
            kp.gridDim.x = blocks;
            gridChanged = true;
        }

        void** params = (void**)kp.kernelParams;
        bool modifiedPtr = false;
        for (int i = 0; i < scanLimit; ++i) {
            void* slot = params[i];
            if (!slot) break;
            if (!hostPtrReadable(slot)) break;
            void* inner = *(void**)slot;

            // 统一匹配三个旧指针
            if (inner == (void*)oldCurr) {
                *(void**)slot = (void*)m_bufs.d_pos_curr;
                modifiedPtr = true;
            }
            else if (inner == (void*)oldNext) {
                *(void**)slot = (void*)m_bufs.d_pos_next;
                modifiedPtr = true;
            }
            else if (inner == (void*)oldPred) {
                *(void**)slot = (void*)m_bufs.d_pos_pred;
                modifiedPtr = true;
            }
        }

        if (modifiedPtr || gridChanged) {
            if (cudaGraphExecKernelNodeSetParams(exec, nd, &kp) == cudaSuccess) {
                if (modifiedPtr) ++patchedPtr;
                if (gridChanged) ++patchedGrid;
            }
        }
    }
    if ((patchedPtr || patchedGrid) && c.debug.printHints) {
        std::fprintf(stderr,
            "[Graph][HotUpdate] PtrPatched=%d GridPatched=%d full=%d\n",
            patchedPtr, patchedGrid, fullGraph ? 1 : 0);
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

// ===== 追加：调试插桩（仅非 Graph 模式） =====
static void DebugPrintParticle0(const char* tag,
    const DeviceBuffers& bufs,
    uint32_t N,
    float dt) {
    if (N == 0) {
        std::fprintf(stderr, "[IntVelDbg][%s] N=0\n", tag);
        return;
    }
    float4 pos0{}, pred0{}, vel0{}, delta0{};
    if (bufs.d_pos_curr) cudaMemcpy(&pos0, bufs.d_pos_curr, sizeof(float4), cudaMemcpyDeviceToHost);
    if (bufs.d_pos_pred) cudaMemcpy(&pred0, bufs.d_pos_pred, sizeof(float4), cudaMemcpyDeviceToHost);
    if (bufs.d_vel)      cudaMemcpy(&vel0, bufs.d_vel, sizeof(float4), cudaMemcpyDeviceToHost);
    if (bufs.d_delta)    cudaMemcpy(&delta0, bufs.d_delta, sizeof(float4), cudaMemcpyDeviceToHost);

    float dx = pred0.x - pos0.x;
    float dy = pred0.y - pos0.y;
    float dz = pred0.z - pos0.z;
    std::fprintf(stderr,
        "[IntVelDbg][%s] pos=(%.6f,%.6f,%.6f) pred=(%.6f,%.6f,%.6f) Δ=(%.6e,%.6e,%.6e) vel=(%.6f,%.6f,%.6f) delta(xsph)=(%.6f,%.6f,%.6f) Δ/dt=(%.6f,%.6f,%.6f) dt=%.6f\n",
        tag,
        pos0.x, pos0.y, pos0.z,
        pred0.x, pred0.y, pred0.z,
        dx, dy, dz,
        vel0.x, vel0.y, vel0.z,
        delta0.x, delta0.y, delta0.z,
        dx / dt, dy / dt, dz / dt, dt);
}

static void DebugPrintVel0(const char* tag, const DeviceBuffers& bufs, uint32_t N, float dt) {
    if (N == 0 || !bufs.d_vel || !bufs.d_pos_curr || !bufs.d_pos_pred) return;
    float4 v0{}, c0{}, p0{};
    cudaMemcpy(&v0, bufs.d_vel, sizeof(float4), cudaMemcpyDeviceToHost);
    cudaMemcpy(&c0, bufs.d_pos_curr, sizeof(float4), cudaMemcpyDeviceToHost);
    cudaMemcpy(&p0, bufs.d_pos_pred, sizeof(float4), cudaMemcpyDeviceToHost);
    float dx = p0.x - c0.x;
    float dy = p0.y - c0.y;
    float dz = p0.z - c0.z;
    std::fprintf(stderr,
        "[PP-Debug][%s][Vel0] v=(%.6f,%.6f,%.6f) predictedΔ=(%.6f,%.6f,%.6f) Δ/dt=(%.6f,%.6f,%.6f) dt=%.6f\n",
        tag, v0.x, v0.y, v0.z, dx, dy, dz, dx / dt, dy / dt, dz / dt, dt);
}

static void DebugCheckNextEqualsPred(const DeviceBuffers& bufs, uint32_t N) {
    if (N == 0 || !bufs.d_pos_next || !bufs.d_pos_pred) return;
    float4 a{}, b{};
    cudaMemcpy(&a, bufs.d_pos_next, sizeof(float4), cudaMemcpyDeviceToHost);
    cudaMemcpy(&b, bufs.d_pos_pred, sizeof(float4), cudaMemcpyDeviceToHost);
    float diff = fabsf(a.x - b.x) + fabsf(a.y - b.y) + fabsf(a.z - b.z);
    std::fprintf(stderr, "[PP-Debug][NextVsPred] diffSumXYZ=%.6e next=%p pred=%p\n", diff, (void*)bufs.d_pos_next, (void*)bufs.d_pos_pred);
}

static void DebugPrintPosDelta(const char* tag,
    const DeviceBuffers& bufs,
    uint32_t N) {
    if (N == 0 || !bufs.d_pos_curr || !bufs.d_pos_pred) {
        std::fprintf(stderr, "[PP-Debug][%s] N=0 or null pointers\n", tag);
        return;
    }
    float4 curr{}, pred{};
    cudaMemcpy(&curr, bufs.d_pos_curr, sizeof(float4), cudaMemcpyDeviceToHost);
    cudaMemcpy(&pred, bufs.d_pos_pred, sizeof(float4), cudaMemcpyDeviceToHost);
    float dx = pred.x - curr.x;
    float dy = pred.y - curr.y;
    float dz = pred.z - curr.z;
    std::fprintf(stderr,
        "[PP-Debug][%s] curr=%p pred=%p Δ=(%.6f,%.6f,%.6f) curr=(%.6f,%.6f,%.6f) pred=(%.6f,%.6f,%.6f)\n",
        tag, (void*)bufs.d_pos_curr, (void*)bufs.d_pos_pred,
        dx, dy, dz,
        curr.x, curr.y, curr.z,
        pred.x, pred.y, pred.z);
}

// ===== Debug: 细粒度打印首粒子位置/速度/预测 =====
static void DebugPrintP0(const char* tag, const DeviceBuffers& bufs, uint32_t N, float dt) {
    if (N == 0) {
        std::fprintf(stderr, "[P0][%s] N=0\n", tag);
        return;
    }
    float4 pos{}, pred{}, vel{}, delta{};
    if (bufs.d_pos_curr) cudaMemcpy(&pos, bufs.d_pos_curr, sizeof(float4), cudaMemcpyDeviceToHost);
    if (bufs.d_pos_pred) cudaMemcpy(&pred, bufs.d_pos_pred, sizeof(float4), cudaMemcpyDeviceToHost);
    if (bufs.d_vel)      cudaMemcpy(&vel, bufs.d_vel, sizeof(float4), cudaMemcpyDeviceToHost);
    if (bufs.d_delta)    cudaMemcpy(&delta, bufs.d_delta, sizeof(float4), cudaMemcpyDeviceToHost);

    float dx = pred.x - pos.x;
    float dy = pred.y - pos.y;
    float dz = pred.z - pos.z;
    std::fprintf(stderr,
        "[P0][%s] pos=(%.6f,%.6f,%.6f) pred=(%.6f,%.6f,%.6f) Δ=(%.6e,%.6e,%.6e) vel=(%.6f,%.6f,%.6f) delta=(%.6f,%.6f,%.6f) Δ/dt=(%.6f,%.6f,%.6f) dt=%.6f\n",
        tag,
        pos.x, pos.y, pos.z,
        pred.x, pred.y, pred.z,
        dx, dy, dz,
        vel.x, vel.y, vel.z,
        delta.x, delta.y, delta.z,
        dx / dt, dy / dt, dz / dt, dt);
}

// ===== Initialization / Shutdown =====
bool Simulator::initialize(const SimParams& p) {
    prof::Range rInit("Sim.Initialize", prof::Color(0x10,0x90,0xF0));

        m_params = p;
        CUDA_CHECK(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));

    const auto& c = console::Instance();
    m_frameTimingEveryN  = c.perf.frame_timing_every_n;
    m_frameTimingEnabled = (m_frameTimingEveryN != 0);
    m_useHashedGrid      = c.perf.use_hashed_grid;

    // Strategy context
    m_ctx.bufs = &m_bufs;
    m_ctx.grid = &m_grid;
    m_ctx.useHashedGrid = m_useHashedGrid;
    if (m_useHashedGrid) {
        m_gridStrategy = std::make_unique<HashedGridStrategy>();
    } else {
        m_gridStrategy = std::make_unique<DenseGridStrategy>();
    }
    m_ctx.gridStrategy = m_gridStrategy.get();
    m_ctx.dispatcher   = &m_kernelDispatcher;

    uint32_t capacity = (p.maxParticles > 0) ? p.maxParticles : p.numParticles;
    if (capacity == 0) capacity = 1;

    bool needHalf = (p.precision.positionStore     == NumericType::FP16_Packed) ||
                    (p.precision.velocityStore     == NumericType::FP16_Packed) ||
                    (p.precision.predictedPosStore == NumericType::FP16_Packed);

    if (needHalf) m_bufs.allocateWithPrecision(p.precision, capacity);
    else          m_bufs.allocate(capacity);

    UpdateDevicePrecisionView(m_bufs, p.precision);

    m_grid.allocateIndices(capacity);
    m_grid.ensureCompactCapacity(capacity);
    if (!buildGrid(m_params)) return false;

    if (capacity > 0) {
        std::vector<uint32_t> h_idx(capacity);
        for (uint32_t i = 0; i < capacity; ++i) h_idx[i] = i;
        CUDA_CHECK(cudaMemcpy(m_grid.d_indices, h_idx.data(), sizeof(uint32_t) * capacity, cudaMemcpyHostToDevice));
    }

    if (p.numParticles > 0) {
        CUDA_CHECK(cudaMemcpy(m_bufs.d_pos_pred, m_bufs.d_pos, sizeof(float4) * p.numParticles, cudaMemcpyDeviceToDevice));
        if (needHalf) m_bufs.packAllToHalf(p.numParticles, m_stream);
    }

    // ===== 半精镜像模式一次性日志（M1） =====
    if (!m_precisionLogged) {
        const auto& pr = m_params.precision;
        bool posHalf  = (pr.positionStore     == NumericType::FP16_Packed);
        bool velHalf  = (pr.velocityStore     == NumericType::FP16_Packed);
        bool predHalf = (pr.predictedPosStore == NumericType::FP16_Packed);
        std::fprintf(stderr,
            "[Precision] Init | posHalf=%d velHalf=%d predHalf=%d | core=%u fp32Acc=%d halfIntr=%d\n",
            posHalf, velHalf, predHalf, (unsigned)pr.coreCompute,
            pr.forceFp32Accumulate ? 1 : 0, pr.enableHalfIntrinsics ? 1 : 0);

        if ((posHalf  && !m_bufs.d_pos_h4) ||
            (velHalf  && !m_bufs.d_vel_h4) ||
            (predHalf && !m_bufs.d_pos_pred_h4)) {
            std::fprintf(stderr,
                "[Precision][Warn] Requested half mirror not allocated (pos=%p vel=%p pred=%p).\n",
                (void*)m_bufs.d_pos_h4, (void*)m_bufs.d_vel_h4, (void*)m_bufs.d_pos_pred_h4);
        } else if (posHalf || velHalf || predHalf) {
            size_t extra = 0;
            if (posHalf)  extra += sizeof(Half4) * m_bufs.capacity;
            if (velHalf)  extra += sizeof(Half4) * m_bufs.capacity;
            if (predHalf) extra += sizeof(Half4) * m_bufs.capacity;
            std::fprintf(stderr,
                "[Precision] Mirror allocated capacity=%u overhead=%.2f MB\n",
                m_bufs.capacity, extra / (1024.0 * 1024.0));
        }

        bool autoMapped = (m_params.useMixedPrecision && posHalf && velHalf && predHalf &&
                            pr.coreCompute == NumericType::FP32 && pr.forceFp32Accumulate);
        std::fprintf(stderr, "[Precision] autoMapMixed=%d\n", autoMapped ? 1 : 0);
        m_precisionLogged = true;
    }

    m_canPingPongPos       = true;
    m_graphDirty           = true;
    m_captured             = {};
    m_cachedNodesReady     = false;
    m_nodeRecycleFull      = nullptr;
    m_nodeRecycleCheap     = nullptr;
    m_lastFrameMs          = -1.f;
    m_evCursor             = 0;
    m_lastParamUpdateFrame = -1;

     

    if (m_pipeline.full().empty()) {
        BuildDefaultPipelines(m_pipeline);
        PostOpsConfig postCfg{}; postCfg.enableXsph = (p.xsph_c > 0.f);
        postCfg.enableBoundary = true; postCfg.enableRecycle = true;
        m_pipeline.post().configure(postCfg, m_useHashedGrid, postCfg.enableXsph);
    }

    m_paramTracker.capture(m_params, m_numCells);
    if (console::Instance().perf.use_cuda_graphs)
    {
		printf("[Simulator] CUDA Graphs enabled.\n");
    }else{
		printf("[Simulator] CUDA Graphs disabled.\n");
    }
    return true;
}

    void Simulator::shutdown() {
        if (m_extPosPred) {
            cudaDestroyExternalMemory(m_extPosPred);
            m_extPosPred = nullptr;
            m_bufs.detachExternalPosPred();
        }

        m_grid.releaseAll();

        if (m_graphExecFull) { cudaGraphExecDestroy(m_graphExecFull);  m_graphExecFull = nullptr; }
        if (m_graphFull) { cudaGraphDestroy(m_graphFull);          m_graphFull = nullptr; }
        if (m_graphExecCheap) { cudaGraphExecDestroy(m_graphExecCheap); m_graphExecCheap = nullptr; }
        if (m_graphCheap) { cudaGraphDestroy(m_graphCheap);         m_graphCheap = nullptr; }

        for (int i = 0; i < 2; ++i) {
            if (m_evStart[i]) { cudaEventDestroy(m_evStart[i]); m_evStart[i] = nullptr; }
            if (m_evEnd[i]) { cudaEventDestroy(m_evEnd[i]);   m_evEnd[i] = nullptr; }
        }

        if (m_stream) { cudaStreamDestroy(m_stream); m_stream = nullptr; }
    }

    // ===== Grid & Param Helpers =====
    bool Simulator::buildGrid(const SimParams& p) {
        int3 dim = GridSystem::ComputeDims(p.grid);
        m_numCells = GridSystem::NumCells(dim);
        if (m_numCells == 0) return false;
        m_grid.allocGridRanges(m_numCells); // fix typo
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

    // ===== Graph Node Caching (Recycle/Velocity updates) =====
    bool Simulator::cacheGraphNodes() {
        m_nodeRecycleFull = nullptr;
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
                        m_nodeRecycleFull = nd;
                        m_kpRecycleBaseFull = kp;
                    }
                    else if (!recordFull && kp.func == target) {
                        m_nodeRecycleCheap = nd;
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

        scan(m_graphFull, m_velNodesFull, m_velNodeParamsBaseFull, m_posNodesFull, m_posNodeParamsBaseFull, true);
        scan(m_graphCheap, m_velNodesCheap, m_velNodeParamsBaseCheap, m_posNodesCheap, m_posNodeParamsBaseCheap, false);

        m_cachedVelNodes = true;
        m_cachedPosNodes = true;
        m_cachedNodesReady = true;
        return true;
    }

    // ===== Legacy Structural / Param Change Detection (kept for now) =====
    bool Simulator::structuralGraphChange(const SimParams& p) const {
        if (!m_graphExecFull || !m_graphExecCheap) return true;
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
        if (structuralGraphChange(p)) return false; // structural takes precedence
        float dtRel = fabsf(p.dt - m_captured.dt) /
            fmaxf(1e-9f, fmaxf(p.dt, m_captured.dt));
        if (dtRel < 0.002f &&
            approxEq3(p.gravity, m_captured.gravity) &&
            approxEq(p.restDensity, m_captured.restDensity) &&
            kernelEqualRelaxed(p.kernel, m_captured.kernel)) {
            return false;
        }
        if (!approxEq(p.dt, m_captured.dt))          return true;
        if (!approxEq3(p.gravity, m_captured.gravity))     return true;
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

        prof::Range r("Sim.GraphCapture", prof::Color(0xE0, 0x60, 0x20));
        GraphBuilder builder;
        auto result = builder.BuildStructural(*this, p);
        return result.structuralRebuilt && result.reuseSucceeded;
    }

    bool Simulator::updateGraphsParams(const SimParams& p) {
        if (!m_paramDirty) return true;
        if (!m_graphExecFull || !m_graphExecCheap) return false;

        const auto& c = console::Instance();

        prof::Range r("Sim.GraphUpdate", prof::Color(0xA0, 0x50, 0x10));
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
        LaunchHashKeysMP(m_grid.d_cellKeys, m_grid.d_indices, m_bufs.d_pos_pred, m_bufs.d_pos_pred_h4, p.grid, p.numParticles, s);
    else
        LaunchHashKeys(m_grid.d_cellKeys, m_grid.d_indices, m_bufs.d_pos_pred, p.grid, p.numParticles, s);
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
        LaunchCellRanges(m_grid.d_cellStart, m_grid.d_cellEnd, m_grid.d_cellKeys_sorted,
            p.numParticles, m_numCells, s);
    }

    void Simulator::kCellRangesCompact(cudaStream_t s, const SimParams& p) {
        prof::Range r("Phase.CellRanges.Dense", prof::Color(0x40, 0xD0, 0xD0));
        LaunchCellRangesCompact(m_grid.d_cellUniqueKeys, m_grid.d_cellOffsets, m_grid.d_compactCount,
            m_grid.d_cellKeys_sorted, p.numParticles, s);
        m_numCompactCells = 0; // placeholder until used
    }

void Simulator::kSolveIter(cudaStream_t s, const SimParams& p) {
    prof::Range r("Phase.SolveIter", prof::Color(0xE0,0x80,0x40));
    DeviceParams dp = MakeDeviceParams(p);

    if (m_useHashedGrid) {
        bool useMP = UseHalfForPosition(p, Stage::LambdaSolve, m_bufs);
        if (useMP) {
            LaunchLambdaCompactMP(m_bufs.d_lambda, m_bufs.d_pos_pred, m_bufs.d_pos_pred_h4,
                                  m_grid.d_indices_sorted, m_grid.d_cellKeys_sorted,
                                  m_grid.d_cellUniqueKeys, m_grid.d_cellOffsets, m_grid.d_compactCount,
                                  dp, p.numParticles, s);
            LaunchDeltaApplyCompactMP(m_bufs.d_pos_pred, m_bufs.d_delta, m_bufs.d_lambda,
                                      m_bufs.d_pos_pred, m_bufs.d_pos_pred_h4,
                                      m_grid.d_indices_sorted, m_grid.d_cellKeys_sorted,
                                      m_grid.d_cellUniqueKeys, m_grid.d_cellOffsets, m_grid.d_compactCount,
                                      dp, p.numParticles, s);
        } else {
            LaunchLambdaCompact(m_bufs.d_lambda, m_bufs.d_pos_pred,
                                 m_grid.d_indices_sorted, m_grid.d_cellKeys_sorted,
                                 m_grid.d_cellUniqueKeys, m_grid.d_cellOffsets, m_grid.d_compactCount,
                                 dp, p.numParticles, s);
            LaunchDeltaApplyCompact(m_bufs.d_pos_pred, m_bufs.d_delta, m_bufs.d_lambda,
                                    m_grid.d_indices_sorted, m_grid.d_cellKeys_sorted,
                                    m_grid.d_cellUniqueKeys, m_grid.d_cellOffsets, m_grid.d_compactCount,
                                    dp, p.numParticles, s);
        }
    } else {
        bool useMP = UseHalfForPosition(p, Stage::LambdaSolve, m_bufs);
        if (useMP) {
            LaunchLambdaMP(m_bufs.d_lambda, m_bufs.d_pos_pred, m_bufs.d_pos_pred_h4,
                           m_grid.d_indices_sorted, m_grid.d_cellKeys_sorted,
                           m_grid.d_cellStart, m_grid.d_cellEnd,
                           dp, p.numParticles, s);
            LaunchDeltaApplyMP(m_bufs.d_pos_pred, m_bufs.d_delta, m_bufs.d_lambda,
                               m_bufs.d_pos_pred, m_bufs.d_pos_pred_h4,
                               m_grid.d_indices_sorted, m_grid.d_cellKeys_sorted,
                               m_grid.d_cellStart, m_grid.d_cellEnd,
                               dp, p.numParticles, s);
        } else {
            LaunchLambda(m_bufs.d_lambda, m_bufs.d_pos_pred,
                         m_grid.d_indices_sorted, m_grid.d_cellKeys_sorted,
                         m_grid.d_cellStart, m_grid.d_cellEnd,
                         dp, p.numParticles, s);
            LaunchDeltaApply(m_bufs.d_pos_pred, m_bufs.d_delta, m_bufs.d_lambda,
                              m_grid.d_indices_sorted, m_grid.d_cellKeys_sorted,
                              m_grid.d_cellStart, m_grid.d_cellEnd,
                              dp, p.numParticles, s);
        }
    }

    LaunchBoundary(m_bufs.d_pos_pred, m_bufs.d_vel, p.grid, 0.0f, p.numParticles, s);
    cudaStreamSynchronize(s);
    DebugPrintPosDelta("AfterSolveIter", m_bufs, p.numParticles);
}

void Simulator::kVelocityAndPost(cudaStream_t s, const SimParams& p) {
    prof::Range r("Phase.VelocityPost", prof::Color(0xC0, 0x40, 0xA0));

    bool semiImplicit = p.pbf.semi_implicit_integration_enable;

    if (!console::Instance().perf.use_cuda_graphs)
        DebugPrintP0("BeforeVelocity", m_bufs, p.numParticles, p.dt);

    bool xsphApplied = false;

    if (!semiImplicit) {
        // 原显式路径：差分推导速度
        bool useMP = UseHalfForPosition(p, Stage::VelocityUpdate, m_bufs);
        if (useMP)
            LaunchVelocityMP(m_bufs.d_vel, m_bufs.d_pos, m_bufs.d_pos_pred,
                m_bufs.d_pos_h4, m_bufs.d_pos_pred_h4,
                1.0f / p.dt, p.numParticles, s);
        else
            LaunchVelocity(m_bufs.d_vel, m_bufs.d_pos, m_bufs.d_pos_pred,
                1.0f / p.dt, p.numParticles, s);
        cudaStreamSynchronize(s);

        if (!console::Instance().perf.use_cuda_graphs)
            DebugPrintP0("AfterVelocity", m_bufs, p.numParticles, p.dt);
    }
    else {
        // 半隐式路径：速度已在积分核中加入重力更新，这里不再差分重写；直接进入 XSPH 以对更新后的 v 做平滑
        if (!console::Instance().perf.use_cuda_graphs)
            DebugPrintP0("SkipVelocityDiff(SemiImplicit)", m_bufs, p.numParticles, p.dt);
    }

    // XSPH 平滑（两种模式都可用）
    if (p.xsph_c > 0.f && p.numParticles > 0) {
        DeviceParams dp = MakeDeviceParams(p);
        bool useMPxs = (UseHalfForPosition(p, Stage::XSPH, m_bufs) &&
            UseHalfForVelocity(p, Stage::XSPH, m_bufs));
        if (m_useHashedGrid) {
            if (useMPxs)
                LaunchXSPHCompactMP(m_bufs.d_delta, m_bufs.d_vel, m_bufs.d_vel_h4,
                    m_bufs.d_pos_pred, m_bufs.d_pos_pred_h4,
                    m_grid.d_indices_sorted, m_grid.d_cellKeys_sorted,
                    m_grid.d_cellUniqueKeys, m_grid.d_cellOffsets, m_grid.d_compactCount,
                    dp, p.numParticles, s);
            else
                LaunchXSPHCompact(m_bufs.d_delta, m_bufs.d_vel, m_bufs.d_pos_pred,
                    m_grid.d_indices_sorted, m_grid.d_cellKeys_sorted,
                    m_grid.d_cellUniqueKeys, m_grid.d_cellOffsets, m_grid.d_compactCount,
                    dp, p.numParticles, s);
        }
        else {
            if (useMPxs)
                LaunchXSPHMP(m_bufs.d_delta, m_bufs.d_vel, m_bufs.d_vel_h4,
                    m_bufs.d_pos_pred, m_bufs.d_pos_pred_h4,
                    m_grid.d_indices_sorted, m_grid.d_cellKeys_sorted,
                    m_grid.d_cellStart, m_grid.d_cellEnd,
                    dp, p.numParticles, s);
            else
                LaunchXSPH(m_bufs.d_delta, m_bufs.d_vel, m_bufs.d_pos_pred,
                    m_grid.d_indices_sorted, m_grid.d_cellKeys_sorted,
                    m_grid.d_cellStart, m_grid.d_cellEnd,
                    dp, p.numParticles, s);
        }
        xsphApplied = true;
        cudaStreamSynchronize(s);
        if (!console::Instance().perf.use_cuda_graphs)
            DebugPrintP0("AfterXSPH", m_bufs, p.numParticles, p.dt);

        // 半隐式与显式都统一：用平滑后的速度覆盖 d_vel
        CUDA_CHECK(cudaMemcpyAsync(m_bufs.d_vel, m_bufs.d_delta,
            sizeof(float4) * p.numParticles,
            cudaMemcpyDeviceToDevice, s));
    }

    float4* effectiveVel = m_bufs.d_vel; // 半隐式：积分使用更新后的速度；显式：差分或 XSPH 后的速度
    LaunchBoundary(m_bufs.d_pos_pred, effectiveVel, p.grid, p.boundaryRestitution, p.numParticles, s);
    LaunchRecycleToNozzleConst(m_bufs.d_pos, m_bufs.d_pos_pred, effectiveVel,
        p.grid, p.dt, p.numParticles, 0, s);
    cudaStreamSynchronize(s);

    if (!console::Instance().perf.use_cuda_graphs)
        DebugPrintP0("AfterBoundaryRecycle", m_bufs, p.numParticles, p.dt);

    if (p.numParticles > 0 && !m_bufs.externalPingPong) {
        CUDA_CHECK(cudaMemcpyAsync(m_bufs.d_pos, m_bufs.d_pos_pred,
            sizeof(float4) * p.numParticles,
            cudaMemcpyDeviceToDevice, s));
    }

    m_ctx.effectiveVel = effectiveVel;
    m_ctx.xsphApplied = xsphApplied;

    if (m_bufs.anyHalf() && m_params.numParticles > 0 && !m_precisionLogged) {
        m_bufs.packAllToHalf(m_params.numParticles, m_stream);
        UpdateDevicePrecisionView(m_bufs, m_params.precision);
        std::fprintf(stderr,
            "[Precision] First emission half-pack done | N=%u | d_pos_h4=%p\n",
            m_params.numParticles, (void*)m_bufs.d_pos_h4);
        m_precisionLogged = true;
    }
    cudaStreamSynchronize(s);
    DebugPrintPosDelta("AfterPostOps", m_bufs, p.numParticles);

    if (!console::Instance().perf.use_cuda_graphs)
        DebugPrintP0("EndVelocityPost", m_bufs, p.numParticles, p.dt);
}

void Simulator::kIntegratePred(cudaStream_t s, const SimParams& p) {
    prof::Range r("Phase.Integrate", prof::Color(0x50, 0xA0, 0xFF));

    if (!console::Instance().perf.use_cuda_graphs)
        DebugPrintP0("BeforeIntegrate", m_bufs, p.numParticles, p.dt);

    // 半隐式模式只使用当前帧累积后的速度（不使用 XSPH effectiveVel，因为那是在后处理阶段才更新）
    bool semiImplicit = p.pbf.semi_implicit_integration_enable;

    const float4* velSrc;
    if (semiImplicit) {
        velSrc = m_bufs.d_vel; // 半隐式：直接用当前速度，内部核会加重力并回写
    }
    else {
        velSrc = (m_ctx.xsphApplied && m_ctx.effectiveVel) ? m_ctx.effectiveVel : m_bufs.d_vel;
    }

    bool useMP = (UseHalfForPosition(p, Stage::Integration, m_bufs) &&
        UseHalfForVelocity(p, Stage::Integration, m_bufs));

    if (semiImplicit) {
        // 调用半隐式核（已经负责回写速度 + 生成 pos_pred）
        LaunchIntegratePredSemiImplicit(m_bufs.d_pos,
            m_bufs.d_vel,      // 非 const
            m_bufs.d_pos_pred,
            p.gravity,
            p.dt,
            p.numParticles,
            s);
    }
    else {
        if (useMP)
            LaunchIntegratePredMP(m_bufs.d_pos, velSrc, m_bufs.d_pos_pred,
                m_bufs.d_pos_h4, m_bufs.d_vel_h4,
                p.gravity, p.dt, p.numParticles, s);
        else
            LaunchIntegratePred(m_bufs.d_pos, velSrc, m_bufs.d_pos_pred,
                p.gravity, p.dt, p.numParticles, s);
    }

    LaunchBoundary(m_bufs.d_pos_pred, m_bufs.d_vel, p.grid, 0.0f, p.numParticles, s);
    cudaStreamSynchronize(s);

    if (!console::Instance().perf.use_cuda_graphs)
        DebugPrintP0("AfterIntegrate", m_bufs, p.numParticles, p.dt);
}

    // ===== Step =====
bool Simulator::step(const SimParams& p) {
    // 不变量：预测缓冲必须与 curr/next 独立
    if (m_bufs.externalPingPong && m_ctx.pingPongPos) {
        bool bad = (m_bufs.d_pos_pred == m_bufs.d_pos_curr) || (m_bufs.d_pos_pred == m_bufs.d_pos_next);
        if (bad && console::Instance().debug.printErrors) {
            std::fprintf(stderr,
                "[ExternalPosPP][Invariant][Fail] pred alias curr/next (pred=%p curr=%p next=%p)\n",
                (void*)m_bufs.d_pos_pred, (void*)m_bufs.d_pos_curr, (void*)m_bufs.d_pos_next);
        }
    }
    prof::Range rf("Sim.Step", prof::Color(0x30, 0x30, 0xA0));
    g_simFrameIndex = m_frameIndex;
    DebugPrintPosDelta("FrameBegin", m_bufs, m_params.numParticles);

    // 参数与发射维护（保持原逻辑）
    m_params = p;
    bool allowPP = console::Instance().perf.allow_pingpong_with_external_pred;
    bool expected = m_canPingPongPos && (!m_bufs.posPredExternal || allowPP);
    if (m_ctx.pingPongPos != expected) {
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
        // 关键：首帧或扩容时让 next 也拥有初始位置，避免首次 swap 后显示空/静止
        if (emitted > 0 && m_bufs.externalPingPong && m_ctx.pingPongPos) {
            CUDA_CHECK(cudaMemcpyAsync(
                m_bufs.d_pos_next,
                m_bufs.d_pos_pred, // 初始播种后 pos_pred 已与 pos 同步
                sizeof(float4) * m_params.numParticles,
                cudaMemcpyDeviceToDevice,
                m_stream));
        }
    }
    if (m_params.numParticles > m_bufs.capacity) {
        const auto& pr = m_params.precision;
        bool needHalf = (pr.positionStore == NumericType::FP16_Packed || pr.positionStore == NumericType::FP16 ||
            pr.velocityStore == NumericType::FP16_Packed || pr.velocityStore == NumericType::FP16 ||
            pr.predictedPosStore == NumericType::FP16_Packed || pr.predictedPosStore == NumericType::FP16);
        if (needHalf) m_bufs.allocateWithPrecision(pr, m_params.numParticles); else m_bufs.allocate(m_params.numParticles);
        UpdateDevicePrecisionView(m_bufs, m_params.precision);
        m_grid.allocateIndices(m_bufs.capacity);
        std::vector<uint32_t> h_idx(m_bufs.capacity);
        for (uint32_t i = 0; i < m_bufs.capacity; ++i) h_idx[i] = i;
        CUDA_CHECK(cudaMemcpy(m_grid.d_indices, h_idx.data(), sizeof(uint32_t) * m_bufs.capacity, cudaMemcpyHostToDevice));
        m_graphDirty = true;
        // 扩容后确保 next 初始填充
        if (m_bufs.externalPingPong && m_ctx.pingPongPos) {
            CUDA_CHECK(cudaMemcpyAsync(
                m_bufs.d_pos_next,
                m_bufs.d_pos_pred,
                sizeof(float4) * m_params.numParticles,
                cudaMemcpyDeviceToDevice,
                m_stream));
        }
    }
    SetEmitParamsAsync(&ep, m_stream);
    updateGridIfNeeded(m_params);

    bool structuralChanged = m_paramTracker.structuralChanged(m_params, m_numCells);
    bool dynamicChanged = (!structuralChanged) && m_paramTracker.dynamicChanged(m_params);
    if (structuralChanged) m_graphDirty = true;
    else if (dynamicChanged) m_paramDirty = true;

    if (m_graphDirty) {
        captureGraphIfNeeded(m_params);
        std::fprintf(stderr, "[Debug] GraphCapture done | d_vel=%p d_delta=%p\n", (void*)m_bufs.d_vel, (void*)m_bufs.d_delta);
    }
    else if (m_paramDirty) updateGraphsParams(m_params);

    int everyN = (console::Instance().perf.sort_compact_every_n <= 0) ? 1 : console::Instance().perf.sort_compact_every_n;
    bool needFull = (m_frameIndex == 0) || (m_lastFullFrame < 0) ||
        ((m_frameIndex - m_lastFullFrame) >= everyN) ||
        (m_params.numParticles != m_captured.numParticles);

    // ===== 新增：在即将启动图之前针对“积分阶段”选择有效速度指针 (下帧积分使用) =====
    bool willUseXsph = (m_params.xsph_c > 0.f && m_params.numParticles > 0);
    // 仅在使用 CUDA Graphs 且节点已缓存时尝试热更新；若结构变化导致重捕获则后续帧再处理
    if (console::Instance().perf.use_cuda_graphs && !m_graphDirty) {
        if (!m_cachedVelNodes) cacheGraphNodes();

        // Full 图与 Cheap 图都各自处理：将需要使用的积分/速度相关节点的 d_vel 替换为 d_delta
        if (willUseXsph) {
            if (!m_velPatchedToDeltaFull) {
                patchGraphVelocityPointers(true, m_bufs.d_vel, m_bufs.d_delta);
                m_velPatchedToDeltaFull = true;
            }
            if (!m_velPatchedToDeltaCheap) {
                patchGraphVelocityPointers(false, m_bufs.d_vel, m_bufs.d_delta);
                m_velPatchedToDeltaCheap = true;
            }
        }
        else { // 需要恢复
            if (m_velPatchedToDeltaFull) {
                patchGraphVelocityPointers(true, m_bufs.d_delta, m_bufs.d_vel);
                m_velPatchedToDeltaFull = false;
            }
            if (m_velPatchedToDeltaCheap) {
                patchGraphVelocityPointers(false, m_bufs.d_delta, m_bufs.d_vel);
                m_velPatchedToDeltaCheap = false;
            }
        }
    }

    // ===== 计时与图执行逻辑（保持原有） =====
    
    bool doTiming = m_frameTimingEnabled && (m_frameTimingEveryN <= 1 ||
        (m_frameIndex % m_frameTimingEveryN) == 0);
    int cur = (m_evCursor & 1);
    int prev = ((m_evCursor + 1) & 1);
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
            cacheGraphNodes();
            if (exec == m_graphExecFull) { node = m_nodeRecycleFull; base = m_kpRecycleBaseFull; }
            else if (exec == m_graphExecCheap) { node = m_nodeRecycleCheap; base = m_kpRecycleBaseCheap; }
            if (node) {
                kp = base; kp.gridDim = dim3(blocks, 1, 1);
                cudaError_t err2 = cudaGraphExecKernelNodeSetParams(exec, node, &kp);
                if (err2 != cudaSuccess) std::fprintf(stderr, "[Graph][Error] Retry setParams failed err=%d (%s)\n", (int)err2, cudaGetErrorString(err2));
            }
        }
        };

    if (c.perf.use_cuda_graphs) {
        if (!m_graphExecFull || !m_graphExecCheap) {
            if (c.debug.printErrors)
                std::fprintf(stderr, "Simulator::step: CUDA Graph not ready.\n");
            return false;
        }
        if (needFull) updNode(m_graphExecFull, m_nodeRecycleFull, m_kpRecycleBaseFull);
        else          updNode(m_graphExecCheap, m_nodeRecycleCheap, m_kpRecycleBaseCheap);
        if (needFull) { CUDA_CHECK(cudaGraphLaunch(m_graphExecFull, m_stream)); m_lastFullFrame = m_frameIndex; }
        else { CUDA_CHECK(cudaGraphLaunch(m_graphExecCheap, m_stream)); }
    }
    else {
        // 非 Graph 路径保持不变
        if (m_pipeline.full().empty()) {
            BuildDefaultPipelines(m_pipeline);
            PostOpsConfig cfg{};
            cfg.enableXsph = (m_params.xsph_c > 0.f);
            cfg.enableBoundary = true;
            cfg.enableRecycle = true;
            m_pipeline.post().configure(cfg, m_useHashedGrid, cfg.enableXsph);
        }
        m_ctx.bufs = &m_bufs;
        m_ctx.grid = &m_grid;
        m_ctx.useHashedGrid = m_useHashedGrid;
        m_ctx.gridStrategy = m_gridStrategy.get();
        m_ctx.dispatcher = &m_kernelDispatcher;
        if (needFull) {
            m_pipeline.runFull(m_ctx, m_params, m_stream);
            m_lastFullFrame = m_frameIndex;
            m_pipeline.runCheap(m_ctx, m_params, m_stream);
        }
        else {
            m_pipeline.runCheap(m_ctx, m_params, m_stream);
        }
    }
    DebugPrintPosDelta("AfterGraphOrPipeline", m_bufs, m_params.numParticles);
    if (m_ctx.pingPongPos && m_bufs.externalPingPong && m_params.numParticles > 0) {
        CUDA_CHECK(cudaMemcpyAsync(
            m_bufs.d_pos_next,
            m_bufs.d_pos_pred,
            sizeof(float4) * m_params.numParticles,
            cudaMemcpyDeviceToDevice,
            m_stream));

        float4* oldCurr = m_bufs.d_pos_curr;
        float4* oldNext = m_bufs.d_pos_next;
        float4* oldPred = nullptr; // 若本帧内预测指针发生过重建，可抓取旧值；这里保持 null 仅在重分配时触发 graphDirty
        std::swap(m_bufs.d_pos_curr, m_bufs.d_pos_next);
        m_bufs.d_pos = m_bufs.d_pos_curr;

        m_swappedThisFrame = true;

        // 若图没被强制重建且使用热更新，则尝试补丁
        if (!m_graphDirty) {
            patchGraphPositionPointers(true, oldCurr, oldNext, oldPred);
            patchGraphPositionPointers(false, oldCurr, oldNext, oldPred);
        }
    }
    signalSimFence(); // signal simulation completion for renderer

    // 计时事件收尾
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
        memDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
        memDesc.handle.win32.handle = sharedHandleWin32;
        memDesc.size = bytes;
        memDesc.flags = cudaExternalMemoryDedicated;
        CUDA_CHECK(cudaImportExternalMemory(&m_extPosPred, &memDesc));

        cudaExternalMemoryBufferDesc bufDesc{};
        bufDesc.offset = 0;
        bufDesc.size = bytes;
        void* devPtr = nullptr;
        CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&devPtr, m_extPosPred, &bufDesc));
        m_bufs.bindExternalPosPred(reinterpret_cast<float4*>(devPtr));
        m_canPingPongPos = true;
        return true;
    }

    bool Simulator::bindExternalPosPingPong(void* sharedHandleA, size_t bytesA, void* sharedHandleB, size_t bytesB) {
        if (!sharedHandleA || !sharedHandleB || bytesA == 0 || bytesB == 0) {
            std::fprintf(stderr, "[ExternalPosPP][Error] invalid handles or sizes A=%p B=%p bytesA=%zu bytesB=%zu\n",
                sharedHandleA, sharedHandleB, bytesA, bytesB);
            return false;
        }
        cudaExternalMemory_t extA = nullptr; cudaExternalMemory_t extB = nullptr;
        cudaExternalMemoryHandleDesc descA{}; descA.type = cudaExternalMemoryHandleTypeD3D12Resource;
        descA.handle.win32.handle = sharedHandleA; descA.size = bytesA; descA.flags = cudaExternalMemoryDedicated;
        cudaExternalMemoryHandleDesc descB{}; descB.type = cudaExternalMemoryHandleTypeD3D12Resource;
        descB.handle.win32.handle = sharedHandleB; descB.size = bytesB; descB.flags = cudaExternalMemoryDedicated;
        if (cudaImportExternalMemory(&extA, &descA) != cudaSuccess) { std::fprintf(stderr, "[ExternalPosPP][Error] import A failed\n"); return false; }
        if (cudaImportExternalMemory(&extB, &descB) != cudaSuccess) { std::fprintf(stderr, "[ExternalPosPP][Error] import B failed\n"); cudaDestroyExternalMemory(extA); return false; }

        cudaExternalMemoryBufferDesc bufA{}; bufA.offset = 0; bufA.size = bytesA; void* devPtrA = nullptr;
        cudaExternalMemoryBufferDesc bufB{}; bufB.offset = 0; bufB.size = bytesB; void* devPtrB = nullptr;
        if (cudaExternalMemoryGetMappedBuffer(&devPtrA, extA, &bufA) != cudaSuccess) {
            std::fprintf(stderr, "[ExternalPosPP][Error] map A failed\n"); cudaDestroyExternalMemory(extA); cudaDestroyExternalMemory(extB); return false;
        }
        if (cudaExternalMemoryGetMappedBuffer(&devPtrB, extB, &bufB) != cudaSuccess) {
            std::fprintf(stderr, "[ExternalPosPP][Error] map B failed\n"); cudaDestroyExternalMemory(extA); cudaDestroyExternalMemory(extB); return false;
        }

        uint32_t capA = static_cast<uint32_t>(bytesA / sizeof(float4));
        uint32_t capB = static_cast<uint32_t>(bytesB / sizeof(float4));
        uint32_t cap = (capA < capB) ? capA : capB;
        if (cap == 0) {
            std::fprintf(stderr, "[ExternalPosPP][Error] capacity zero after import\n");
            cudaDestroyExternalMemory(extA); cudaDestroyExternalMemory(extB);
            return false;
        }

        // 保存 A 句柄；B 后续可加入管理，这里保持与原逻辑一致
        if (m_extPosPred) { cudaDestroyExternalMemory(m_extPosPred); m_extPosPred = nullptr; }
        m_extPosPred = extA;
        m_extraExternalMemB = extB; // 新增：可在 shutdown 中销毁（需在类中声明 cudaExternalMemory_t m_extraExternalMemB = nullptr;)

        // 保存旧预测指针
        float4* oldPredPtr = m_bufs.d_pos_pred;

        // 仅绑定 curr/next，不动内部 d_pos_pred
        m_bufs.bindExternalPosPingPong(reinterpret_cast<float4*>(devPtrA),
            reinterpret_cast<float4*>(devPtrB), cap);

        if (m_bufs.d_pos_pred == m_bufs.d_pos_curr || m_bufs.d_pos_pred == m_bufs.d_pos_next) {
            float4* newPred = nullptr;
            CUDA_CHECK(cudaMalloc(&newPred, sizeof(float4) * cap));
            m_bufs.d_pos_pred = newPred;
            CUDA_CHECK(cudaMemcpy(m_bufs.d_pos_pred, m_bufs.d_pos_curr,
                sizeof(float4) * cap, cudaMemcpyDeviceToDevice));
            std::fprintf(stderr, "[ExternalPosPP][Fix] Reallocated internal d_pos_pred=%p (cap=%u)\n",
                (void*)m_bufs.d_pos_pred, cap);

            // 关键：标记 graph 需要结构重建（旧图中所有 kernel 还引用 oldPredPtr）
            m_graphDirty = true;
        }

        m_canPingPongPos = true;
        m_ctx.pingPongPos = true;
        std::fprintf(stderr, "[ExternalPosPP][Ready] curr=%p next=%p pred=%p cap=%u\n",
            (void*)m_bufs.d_pos_curr, (void*)m_bufs.d_pos_next, (void*)m_bufs.d_pos_pred, cap);
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
                for (;;) { ox = U(jrng); oy = U(jrng); oz = U(jrng); if (ox * ox + oy * oy + oz * oz <= 1.f) break; }
                ox *= jitterAmp; oy *= jitterAmp; oz *= jitterAmp;
                float4 p4 = h_pos[i];
                p4.x = fminf(fmaxf(p4.x + ox, m_params.grid.mins.x), m_params.grid.maxs.x);
                p4.y = fminf(fmaxf(p4.y + oy, m_params.grid.mins.y), m_params.grid.maxs.y);
                p4.z = fminf(fmaxf(p4.z + oz, m_params.grid.mins.z), m_params.grid.maxs.z);
                h_pos[i] = p4;
            }
        }

        CUDA_CHECK(cudaMemcpy(m_bufs.d_pos, h_pos.data(), sizeof(float4) * total, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m_bufs.d_pos_pred, h_pos.data(), sizeof(float4) * total, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m_bufs.d_vel, h_vel.data(), sizeof(float4) * total, cudaMemcpyHostToDevice));
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
        double avgN = 0.0, avgV = 0.0, avgRhoRel_g = 0.0, avgR_g = 0.0;
        uint32_t stride = (sampleStride == 0 ? 1u : sampleStride);
        bool ok = LaunchComputeStats(m_bufs.d_pos_pred, m_bufs.d_vel,
            m_grid.d_indices_sorted, m_grid.d_cellStart, m_grid.d_cellEnd,
            m_params.grid, m_params.kernel, m_params.particleMass,
            m_params.numParticles, m_numCells, stride,
            &avgN, &avgV, &avgRhoRel_g, &avgR_g, m_stream); // fix arg name
        if (!ok) return false;
        out.N = m_params.numParticles;
        out.avgNeighbors = avgN;
        out.avgSpeed = avgV;
        out.avgRho = avgR_g;
        out.avgRhoRel = (m_params.restDensity > 0.f) ? (avgR_g / (double)m_params.restDensity) : 0.0;
        bool wantDiag = (c.debug.printDiagnostics || c.debug.printWarnings || c.debug.printHints);
        uint64_t diagEvery = (uint64_t)(c.debug.logEveryN <= 0 ? 1 : c.debug.logEveryN);
        if (!wantDiag || (m_frameIndex % diagEvery) != 0ull) return true;
        double avgN_bf = 0.0, avgV_bf = 0.0, avgRhoRel_bf = 0.0, avgR_bf = 0.0; const uint32_t kMaxISamples = 2048;
        bool ok_bf = LaunchComputeStatsBruteforce(m_bufs.d_pos_pred, m_bufs.d_vel, m_params.kernel, m_params.particleMass,
            m_params.numParticles, stride, kMaxISamples,
            &avgN_bf, &avgV_bf, &avgRhoRel_bf, &avgR_bf, m_stream);
        if (!ok_bf) return true; // ignore diag failure
        bool hasCap = (m_params.maxNeighbors > 0); double capN = hasCap ? double(m_params.maxNeighbors) : std::numeric_limits<double>::infinity();
        bool nearCap = hasCap && (avgN >= 0.9 * capN);
        bool severeUnder = (avgN_bf > 0.0) && (avgN < 0.8 * avgN_bf);
        double rhoRel_bf = (m_params.restDensity > 0.f) ? (avgR_bf / (double)m_params.restDensity) : 0.0;
        static uint64_t s_last = UINT64_MAX; bool shouldDiag = (nearCap || severeUnder);
        if (shouldDiag && s_last != m_frameIndex && wantDiag) {
            s_last = m_frameIndex; double h = double(m_params.kernel.h); double cell = double(m_params.grid.cellSize); double ratio = (h > 0.0) ? (cell / h) : 0.0; if (c.debug.printDiagnostics) {
                std::fprintf(stderr,
                    "[Diag] Frame=%llu | N=%u | h=%.6g | cell=%.3f (cell/h=%.3f) | dim=(%d,%d,%d) numCells=%u\n",
                    (unsigned long long)m_frameIndex, m_params.numParticles, h, cell, ratio,
                    m_params.grid.dim.x, m_params.grid.dim.y, m_params.grid.dim.z, m_numCells);
                std::fprintf(stderr,
                    "[Diag] Neighbors: grid=%.3f, brute=%.3f | RhoRel: grid=%.3f, brute=%.3f | maxNeighbors=%d%s\n",
                    avgN, avgN_bf, out.avgRhoRel, rhoRel_bf, m_params.maxNeighbors, (nearCap ? " (near cap)" : ""));
            }
            if (c.debug.printHints) { if (ratio < 0.9) std::fprintf(stderr, "[Hint] cellSize/h <0.9 建议 ~[1.0,1.5]h\n"); else if (ratio > 2.0) std::fprintf(stderr, "[Hint] cellSize/h >2.0 建议 ~[1.0,1.5]h\n"); if (nearCap) std::fprintf(stderr, "[Hint] avgNeighbors 接近上限 %d\n", m_params.maxNeighbors); if (severeUnder) std::fprintf(stderr, "[Hint] 网格邻居显著偏低\n"); }
        }
        return true;
    }

    bool Simulator::computeStatsBruteforce(SimStats& out, uint32_t sampleStride, uint32_t maxISamples) const {
        if (m_params.numParticles == 0) { out = {}; return true; }
        double avgN = 0.0, avgV = 0.0, avgRhoRel = 0.0, avgR = 0.0;
        bool ok = LaunchComputeStatsBruteforce(m_bufs.d_pos_pred, m_bufs.d_vel, m_params.kernel, m_params.particleMass,
            m_params.numParticles, (sampleStride == 0 ? 1u : sampleStride), maxISamples,
            &avgN, &avgV, &avgRhoRel, &avgR, m_stream);
        if (!ok) return false;
        out.N = m_params.numParticles;
        out.avgNeighbors = avgN;
        out.avgSpeed = avgV;
        out.avgRho = avgR;
        out.avgRhoRel = (m_params.restDensity > 0.f) ? (avgR / (double)m_params.restDensity) : 0.0;
        return true;
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
        m_simFenceValue = 0; // start at 0; renderer will wait >0 values
        return true;
    }

    void Simulator::signalSimFence(){
        if(!m_extTimelineSem) return;
        ++m_simFenceValue; // increment simulation completion value
        cudaExternalSemaphoreSignalParams params{}; params.params.fence.value = m_simFenceValue; params.flags = 0;
        cudaSignalExternalSemaphoresAsync(&m_extTimelineSem,&params,1,m_stream);
    }

    void Simulator::patchGraphVelocityPointers(bool fullGraph,
        const float4* fromPtr,
        const float4* toPtr) {
        const auto& c = console::Instance();
        if (!c.perf.graph_hot_update_enable) return;
        if (!fromPtr || !toPtr || fromPtr == toPtr) return;

        cudaGraphExec_t exec = fullGraph ? m_graphExecFull : m_graphExecCheap;
        if (!exec) return;

        // 确保节点缓存已建立
        if (!m_cachedVelNodes || (fullGraph ? m_velNodesFull.empty() : m_velNodesCheap.empty())) return;

        auto& nodes = fullGraph ? m_velNodesFull : m_velNodesCheap;

        int userLimit = c.perf.graph_hot_update_scan_limit;
        int scanLimit = (userLimit > 0 && userLimit <= 4096) ? userLimit : 256;

        int patched = 0;
        for (auto nd : nodes) {
            cudaKernelNodeParams kp{};
            if (cudaGraphKernelNodeGetParams(nd, &kp) != cudaSuccess) continue;
            if (!kp.kernelParams) continue;

            bool modified = false;
            void** params = (void**)kp.kernelParams;
            for (int i = 0; i < scanLimit; ++i) {
                void* slot = params[i];
                if (!slot) break;
                if (!hostPtrReadable(slot)) break;
                void* inner = *(void**)slot;
                if (inner == (void*)fromPtr) {
                    *(void**)slot = (void*)toPtr;
                    modified = true;
                }
            }
            if (modified) {
                if (cudaGraphExecKernelNodeSetParams(exec, nd, &kp) == cudaSuccess) {
                    ++patched;
                }
            }
        }
        if (patched && c.debug.printHints) {
            std::fprintf(stderr,
                "[Graph][HotUpdate][Vel] full=%d from=%p to=%p patched=%d\n",
                fullGraph ? 1 : 0, (const void*)fromPtr, (const void*)toPtr, patched);
        }
    }
} // namespace sim