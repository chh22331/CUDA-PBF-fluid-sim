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
    uint64_t g_simFrameIndex = 0;

    // 新增：渲染半精外部缓冲导入
    bool Simulator::importRenderHalfBuffer(void* sharedHandleWin32, size_t bytes) {
        if (!sharedHandleWin32 || bytes ==0) return false;
        if (m_extRenderHalf) {
            cudaDestroyExternalMemory(m_extRenderHalf);
            m_extRenderHalf = nullptr;
            m_renderHalfMappedPtr = nullptr;
            m_renderHalfBytes =0;
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
        cudaExternalMemoryBufferDesc bufDesc{}; bufDesc.offset =0; bufDesc.size = bytes;
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
        if (!m_renderHalfMappedPtr || !m_extRenderHalf || count ==0) return;
        if (!m_bufs.useRenderHalf || !m_bufs.d_render_pos_h4) return;
        size_t needBytes = size_t(count) * sizeof(sim::Half4);
        if (needBytes > m_renderHalfBytes) {
            std::fprintf(stderr, "[RenderHalf][Warn] external buffer too small need=%zu have=%zu\n", needBytes, m_renderHalfBytes); return; }
        if (m_bufs.nativeHalfActive) {
            // 原生 half 主存储：直接从当前 half4 指针复制（curr）
            CUDA_LOGGED_MEMCPY_D2D_ASYNC("RenderHalf.Publish.Native", m_renderHalfMappedPtr, m_bufs.d_pos_h4, needBytes, m_stream);
        } else {
            // 非原生：需要先 pack 当前 curr 到渲染镜像
            m_bufs.packRenderToHalf(count, m_stream);
            CUDA_LOGGED_MEMCPY_D2D_ASYNC("RenderHalf.Publish", m_renderHalfMappedPtr, m_bufs.d_render_pos_h4, needBytes, m_stream);
        }
    }

    void Simulator::releaseRenderHalfExternal() {
        if (m_extRenderHalf) {
            cudaDestroyExternalMemory(m_extRenderHalf);
            m_extRenderHalf = nullptr;
        }
        m_renderHalfMappedPtr = nullptr;
        m_renderHalfBytes =0;
    }

    // Add helper to patch kernel node params with new position pointers
    void sim::Simulator::patchGraphPositionPointers(bool fullGraph,
        float4* oldCurr,
        float4* oldNext,
        float4* oldPred)
    {
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

            // 动态更新 gridDim（粒子数变化）
            uint32_t blocks = (m_params.numParticles + 255u) / 256u;
            if (blocks == 0) blocks = 1;
            if (kp.gridDim.x != blocks) {
                kp.gridDim.x = blocks;
                ++patchedGrid;
            }

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
                } else if (inner == oldNext || inner == oldPred) {
                    *(void**)slot = (void*)m_bufs.d_pos_next; // pred 统一映射到 next
                    modified = true;
                }
            }
            if (modified || patchedGrid) {
                if (cudaGraphExecKernelNodeSetParams(exec, nd, &kp) == cudaSuccess) {
                    if (modified) ++patchedPtr;
                }
            }
        }

        if ((patchedPtr || patchedGrid) && c.debug.printHints) {
            std::fprintf(stderr,
                "[Graph][PosPatch] full=%d patchedPtr=%d patchedGrid=%d curr=%p next=%p\n",
                fullGraph ? 1 : 0, patchedPtr, patchedGrid,
                (void*)m_bufs.d_pos_curr, (void*)m_bufs.d_pos_next);
        }
        (void)fullGraph; (void)oldCurr; (void)oldNext; (void)oldPred;
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

        bool needHalf = (p.precision.positionStore == NumericType::FP16_Packed) ||
                        (p.precision.velocityStore == NumericType::FP16_Packed) ||
                        (p.precision.predictedPosStore == NumericType::FP16_Packed) ||
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

        // 初始：curr -> next，pred 设为 next（alias）
        if (p.numParticles > 0) {
            CUDA_CHECK(cudaMemcpy(m_bufs.d_pos_next, m_bufs.d_pos,
                                  sizeof(float4) * p.numParticles,
                                  cudaMemcpyDeviceToDevice));
            m_bufs.d_pos_pred = m_bufs.d_pos_next;
            if (needHalf) m_bufs.packAllToHalf(p.numParticles, m_stream);
        }
        UploadSimPosTableConst(m_bufs.d_pos_curr, m_bufs.d_pos_next);

        m_canPingPongPos = true;
        m_graphDirty = true;
        m_captured = {};
        m_cachedNodesReady = false;
        m_nodeRecycleFull = nullptr;
        m_nodeRecycleCheap = nullptr;
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

        if (m_extRenderHalf) {
            cudaDestroyExternalMemory(m_extRenderHalf);
            m_extRenderHalf = nullptr;
        }

        m_grid.releaseAll();

        if (m_graphExecFull)  { cudaGraphExecDestroy(m_graphExecFull);  m_graphExecFull = nullptr; }
        if (m_graphFull)      { cudaGraphDestroy(m_graphFull);          m_graphFull = nullptr; }
        if (m_graphExecCheap) { cudaGraphExecDestroy(m_graphExecCheap); m_graphExecCheap = nullptr; }
        if (m_graphCheap)     { cudaGraphDestroy(m_graphCheap);         m_graphCheap = nullptr; }

        for (int i = 0; i < 2; ++i) {
            if (m_evStart[i]) { cudaEventDestroy(m_evStart[i]); m_evStart[i] = nullptr; }
            if (m_evEnd[i])   { cudaEventDestroy(m_evEnd[i]);   m_evEnd[i] = nullptr; }
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
                } else if (!recordFull && kp.func == target) {
                    m_nodeRecycleCheap = nd;
                    m_kpRecycleBaseCheap = kp;
                }

                if (!kp.kernelParams) continue;
                void** params = (void**)kp.kernelParams;
                std::unordered_set<const void*> watchVel{ (const void*)m_bufs.d_vel, (const void*)m_bufs.d_delta };
                std::unordered_set<const void*> watchPos; // 支持原生 half 模式
                if (m_bufs.nativeHalfActive) {
                    if (m_bufs.d_pos_h4) watchPos.insert((const void*)m_bufs.d_pos_h4);
                    if (m_bufs.d_pos_pred_h4) watchPos.insert((const void*)m_bufs.d_pos_pred_h4);
                } else {
                    if (m_bufs.d_pos_curr) watchPos.insert((const void*)m_bufs.d_pos_curr);
                    if (m_bufs.d_pos_next) watchPos.insert((const void*)m_bufs.d_pos_next);
                    if (m_bufs.d_pos) watchPos.insert((const void*)m_bufs.d_pos);
                    if (m_bufs.d_pos_pred) watchPos.insert((const void*)m_bufs.d_pos_pred);
                }
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

        m_cachedVelNodes = true;
        m_cachedPosNodes = true;
        m_cachedNodesReady = true;
        return true;
    }

    // ===== Graph 变化检测 =====
    bool Simulator::structuralGraphChange(const SimParams& p) const {
        if (!m_graphExecFull || !m_graphExecCheap) return true;
        if (p.solverIters  != m_captured.solverIters)   return true;
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
        prof::Range r("Phase.HashKeys", prof::Color(0x60, 0xB0, 0x40));
        bool useMP = UseHalfForPosition(p, Stage::GridBuild, m_bufs);
        if (useMP)
            LaunchHashKeysMP(m_grid.d_cellKeys, m_grid.d_indices,
                             m_bufs.d_pos_next, m_bufs.d_pos_pred_h4,
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

    void Simulator::kCellRangesCompact(cudaStream_t s, const SimParams& p) {
        prof::Range r("Phase.CellRanges.Dense", prof::Color(0x40, 0xD0, 0xD0));
        LaunchCellRangesCompact(m_grid.d_cellUniqueKeys, m_grid.d_cellOffsets,
                                m_grid.d_compactCount,
                                m_grid.d_cellKeys_sorted, p.numParticles, s);
        m_numCompactCells = 0;
    }

    void Simulator::kSolveIter(cudaStream_t s, const SimParams& p) {
        prof::Range r("Phase.SolveIter", prof::Color(0xE0, 0x80, 0x40));
        DeviceParams dp = MakeDeviceParams(p);

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

        LaunchBoundary(m_bufs.d_pos_next, m_bufs.d_vel, p.grid, 0.0f, p.numParticles, s);
    }
    void Simulator::kVelocityAndPost(cudaStream_t s, const SimParams& p) {
        prof::Range r("Phase.VelocityPost", prof::Color(0xC0, 0x40, 0xA0));

        bool useMP = UseHalfForPosition(p, Stage::VelocityUpdate, m_bufs);
        if (useMP)
            LaunchVelocityMP(m_bufs.d_vel, m_bufs.d_pos, m_bufs.d_pos_next,
                             m_bufs.d_pos_h4, m_bufs.d_pos_pred_h4,
                             1.0f / p.dt, p.numParticles, s);
        else
            LaunchVelocity(m_bufs.d_vel, m_bufs.d_pos, m_bufs.d_pos_next,
                           1.0f / p.dt, p.numParticles, s);

        bool xsphApplied = false;
        if (p.xsph_c > 0.f && p.numParticles > 0) {
            DeviceParams dp = MakeDeviceParams(p);
            bool useMPxs = (UseHalfForPosition(p, Stage::XSPH, m_bufs) &&
                            UseHalfForVelocity(p, Stage::XSPH, m_bufs));
            if (m_useHashedGrid) {
                if (useMPxs)
                    LaunchXSPHCompactMP(m_bufs.d_delta, m_bufs.d_vel, m_bufs.d_vel_h4,
                                        m_bufs.d_pos_next, m_bufs.d_pos_pred_h4,
                                        m_grid.d_indices_sorted, m_grid.d_cellKeys_sorted,
                                        m_grid.d_cellUniqueKeys, m_grid.d_cellOffsets, m_grid.d_compactCount,
                                        dp, p.numParticles, s);
                else
                    LaunchXSPHCompact(m_bufs.d_delta, m_bufs.d_vel,
                                      m_bufs.d_pos_next,
                                      m_grid.d_indices_sorted, m_grid.d_cellKeys_sorted,
                                      m_grid.d_cellUniqueKeys, m_grid.d_cellOffsets, m_grid.d_compactCount,
                                      dp, p.numParticles, s);
            } else {
                if (useMPxs)
                    LaunchXSPHMP(m_bufs.d_delta, m_bufs.d_vel, m_bufs.d_vel_h4,
                                 m_bufs.d_pos_next, m_bufs.d_pos_pred_h4,
                                 m_grid.d_indices_sorted, m_grid.d_cellKeys_sorted,
                                 m_grid.d_cellStart, m_grid.d_cellEnd,
                                 dp, p.numParticles, s);
                else
                    LaunchXSPH(m_bufs.d_delta, m_bufs.d_vel, m_bufs.d_pos_next,
                               m_grid.d_indices_sorted, m_grid.d_cellKeys_sorted,
                               m_grid.d_cellStart, m_grid.d_cellEnd,
                               dp, p.numParticles, s);
            }
            xsphApplied = true;
        }

        float4* effectiveVel = xsphApplied ? m_bufs.d_delta : m_bufs.d_vel;
        LaunchBoundary(m_bufs.d_pos_next, effectiveVel, p.grid, p.boundaryRestitution, p.numParticles, s);
        LaunchRecycleToNozzleConst(m_bufs.d_pos, m_bufs.d_pos_next, effectiveVel,
                                   p.grid, p.dt, p.numParticles, 0, s);

        cudaStreamSynchronize(s);
    }

    void Simulator::kIntegratePred(cudaStream_t s, const SimParams& p) {
        prof::Range r("Phase.Integrate", prof::Color(0x50, 0xA0, 0xFF));

        const float4* velSrc = (m_ctx.xsphApplied && m_ctx.effectiveVel)
            ? m_ctx.effectiveVel : m_bufs.d_vel;

        bool useMP = (UseHalfForPosition(p, Stage::Integration, m_bufs) &&
                      UseHalfForVelocity(p, Stage::Integration, m_bufs));

        if (useMP)
            LaunchIntegratePredMP(m_bufs.d_pos, velSrc, m_bufs.d_pos_next,
                                  m_bufs.d_pos_h4, m_bufs.d_vel_h4,
                                  p.gravity, p.dt, p.numParticles, s);
        else
            LaunchIntegratePred(m_bufs.d_pos, velSrc, m_bufs.d_pos_next,
                                p.gravity, p.dt, p.numParticles, s);

        LaunchBoundary(m_bufs.d_pos_next, m_bufs.d_vel, p.grid, 0.0f, p.numParticles, s);

        cudaStreamSynchronize(s);
    }

    // ===== Step =====
    bool Simulator::step(const SimParams& p) {
        prof::Range rf("Sim.Step", prof::Color(0x30, 0x30, 0xA0));
        g_simFrameIndex = m_frameIndex;
        m_swappedThisFrame = false;

        m_params = p;

        bool allowPP = console::Instance().perf.allow_pingpong_with_external_pred;
        bool expected = m_canPingPongPos && (!m_bufs.posPredExternal || allowPP);
        if (m_ctx.pingPongPos != expected)
            m_ctx.pingPongPos = expected;

        const auto& cHot = console::Instance();
        if (m_frameTimingEveryN != cHot.perf.frame_timing_every_n) {
            m_frameTimingEveryN = cHot.perf.frame_timing_every_n;
            m_frameTimingEnabled = (m_frameTimingEveryN != 0);
        }

        // 发射
        EmitParams ep{};
        console::BuildEmitParams(console::Instance(), ep);
        uint32_t capacity = m_bufs.capacity;
        if (m_params.maxParticles == 0) m_params.maxParticles = capacity;
        m_params.maxParticles = std::min<uint32_t>(m_params.maxParticles, capacity);

        uint32_t emitted = Emitter::EmitFaucet(m_bufs, m_params,
            console::Instance(), ep, m_frameIndex, m_stream);

        if (emitted > 0 && m_bufs.anyHalf())
            m_bufs.packAllToHalf(m_params.numParticles, m_stream);

        // 扩容
        if (m_params.numParticles > m_bufs.capacity) {
            const auto& pr = m_params.precision;
            bool needHalf2 = (pr.positionStore == NumericType::FP16_Packed ||
                              pr.velocityStore == NumericType::FP16_Packed ||
                              pr.predictedPosStore == NumericType::FP16_Packed ||
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
            UploadSimPosTableConst(m_bufs.d_pos_curr, m_bufs.d_pos_next); // [SchemeC] refresh
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

        int everyN = (console::Instance().perf.sort_compact_every_n <= 0)
            ? 1 : console::Instance().perf.sort_compact_every_n;
        bool needFull = (m_frameIndex == 0) || (m_lastFullFrame < 0) ||
                        ((m_frameIndex - m_lastFullFrame) >= everyN) ||
                        (m_params.numParticles != m_captured.numParticles);

        // 计时事件
        bool doTiming = m_frameTimingEnabled &&
                        (m_frameTimingEveryN <= 1 ||
                         (m_frameIndex % m_frameTimingEveryN) == 0);

        int cur  = (m_evCursor & 1);
        int prev = ((m_evCursor + 1) & 1);
        if (doTiming) {
            for (int i = 0; i < 2; ++i) {
                if (!m_evStart[i]) cudaEventCreate(&m_evStart[i]);
                if (!m_evEnd[i])   cudaEventCreate(&m_evEnd[i]);
            }
            if (m_evStart[cur] && m_evEnd[cur])
                CUDA_CHECK(cudaEventRecord(m_evStart[cur], m_stream));
        }

        // kernel node recycle 更新
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
                if (exec == m_graphExecFull) {
                    node = m_nodeRecycleFull;  base = m_kpRecycleBaseFull;
                } else if (exec == m_graphExecCheap) {
                    node = m_nodeRecycleCheap; base = m_kpRecycleBaseCheap;
                }
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

        // 执行
        if (console::Instance().perf.use_cuda_graphs) {
            if (!m_graphExecFull || !m_graphExecCheap) {
                if (console::Instance().debug.printErrors)
                    std::fprintf(stderr, "[Step][Error] Graph not ready.\n");
                return false;
            }
            if (needFull) updNode(m_graphExecFull, m_nodeRecycleFull, m_kpRecycleBaseFull);
            else          updNode(m_graphExecCheap, m_nodeRecycleCheap, m_kpRecycleBaseCheap);

            if (needFull) {
                CUDA_CHECK(cudaGraphLaunch(m_graphExecFull, m_stream));
                m_lastFullFrame = m_frameIndex;
            } else {
                CUDA_CHECK(cudaGraphLaunch(m_graphExecCheap, m_stream));
            }
        } else {
            // 非 Graph 路径
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
            } else {
                m_pipeline.runCheap(m_ctx, m_params, m_stream);
            }
        }

        // 帧尾：ping-pong 交换，无 Pred->Next 拷贝
        if (m_ctx.pingPongPos && m_bufs.externalPingPong && m_params.numParticles >0) {
            float4* oldCurr = m_bufs.d_pos_curr;
            float4* oldNext = m_bufs.d_pos_next;
            float4* oldPred = m_bufs.d_pos_pred; // alias

            m_bufs.swapPositionPingPong();
            m_swappedThisFrame = true;
            m_bufs.d_pos_pred = m_bufs.d_pos_next;
            UploadSimPosTableConst(m_bufs.d_pos_curr, m_bufs.d_pos_next);
            if (console::Instance().perf.use_cuda_graphs && console::Instance().perf.graph_hot_update_enable) {
                patchGraphPositionPointers(true, oldCurr, oldNext, oldPred);
                patchGraphPositionPointers(false, oldCurr, oldNext, oldPred);
            }
        } else if (m_bufs.nativeHalfActive && m_ctx.pingPongPos && m_params.numParticles >0) {
            // 原生 half 主存储 ping-pong（只交换 half4 指针）
            sim::Half4* oldCurrH = m_bufs.d_pos_h4; // 当前别名
            sim::Half4* oldNextH = m_bufs.d_pos_pred_h4;
            m_bufs.swapPositionPingPong();
            m_swappedThisFrame = true;
            UploadSimPosTableConst((float4*)nullptr, (float4*)nullptr); // FP32 常量表无效，但仍调用以保持流程
            if (console::Instance().perf.use_cuda_graphs && console::Instance().perf.graph_hot_update_enable) {
                patchGraphHalfPositionPointers(true, oldCurrH, oldNextH);
                patchGraphHalfPositionPointers(false, oldCurrH, oldNextH);
            }
        } else {
            if (console::Instance().debug.printHints) {
                std::fprintf(stderr,
                    "[PingPong][Skip] frame=%llu ext=%d can=%d ping=%d\n",
                    (unsigned long long)m_frameIndex,
                    (int)m_bufs.externalPingPong,
                    (int)m_canPingPongPos,
                    (int)m_ctx.pingPongPos);
            }
        }

        signalSimFence();
        // 新增：同步幽灵粒子计数到设备常量（若使用）
        if(p.ghostParticleCount !=0){ UploadGhostCount(p.ghostParticleCount); }

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
        // 渲染半精发布：在 fence signal 后。用户需保证导入外部 half buffer。
        if(m_params.precision.renderTransfer == NumericType::FP16_Packed || m_params.precision.renderTransfer == NumericType::FP16){ publishRenderHalf(m_params.numParticles); }
        debugLogPredictedPosHalf(m_params);
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
            (void*)m_bufs.d_pos_pred, cap, N);

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
        if (m_bufs.anyHalf()) m_bufs.packAllToHalf(total, 0);
    }

    bool Simulator::computeStats(SimStats& out, uint32_t sampleStride) const {
        out = {};
        if (m_params.numParticles == 0 || m_bufs.capacity == 0) return true;
        if (m_useHashedGrid) {
            LaunchCellRanges(m_grid.d_cellStart, m_grid.d_cellEnd,
                             m_grid.d_cellKeys_sorted,
                             m_params.numParticles, m_numCells, m_stream);
            CUDA_CHECK(cudaStreamSynchronize(m_stream));
        }
        double avgN = 0.0, avgV = 0.0, avgRhoRel_g = 0.0, avgR_g = 0.0;
        uint32_t stride = (sampleStride == 0 ? 1u : sampleStride);
        bool ok = LaunchComputeStats(m_bufs.d_pos_next, m_bufs.d_vel,
                                     m_grid.d_indices_sorted,
                                     m_grid.d_cellStart, m_grid.d_cellEnd,
                                     m_params.grid, m_params.kernel, m_params.particleMass,
                                     m_params.numParticles, m_numCells, stride,
                                     &avgN, &avgV, &avgRhoRel_g, &avgR_g, m_stream);
        if (!ok) return false;
        out.N = m_params.numParticles;
        out.avgNeighbors = avgN;
        out.avgSpeed = avgV;
        out.avgRho = avgR_g;
        out.avgRhoRel = (m_params.restDensity > 0.f) ? (avgR_g / (double)m_params.restDensity) : 0.0;
        return true;
    }

    bool Simulator::computeStatsBruteforce(SimStats& out, uint32_t sampleStride, uint32_t maxISamples) const {
        if (m_params.numParticles == 0) { out = {}; return true; }
        double avgN = 0.0, avgV = 0.0, avgRhoRel = 0.0, avgR = 0.0;
        bool ok = LaunchComputeStatsBruteforce(m_bufs.d_pos_next, m_bufs.d_vel,
                                               m_params.kernel, m_params.particleMass,
                                               m_params.numParticles, (sampleStride == 0 ? 1u : sampleStride),
                                               maxISamples, &avgN, &avgV, &avgRhoRel, &avgR, m_stream);
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
        m_simFenceValue = 0;
        return true;
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
    void Simulator::patchGraphVelocityPointers(bool fullGraph,
        const float4* fromPtr,
        const float4* toPtr)
    {
        const auto& c = console::Instance();
        if (!c.perf.graph_hot_update_enable) return;
        if (!fromPtr || !toPtr || fromPtr == toPtr) return;

        cudaGraphExec_t exec = fullGraph ? m_graphExecFull : m_graphExecCheap;
        if (!exec) return;

        if (!m_cachedVelNodes || (fullGraph ? m_velNodesFull.empty() : m_velNodesCheap.empty()))
            return;

        auto& nodes = fullGraph ? m_velNodesFull : m_velNodesCheap;
        int userLimit = c.perf.graph_hot_update_scan_limit;
        int scanLimit = (userLimit > 0 && userLimit <= 4096) ? userLimit : 256;

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
                if (inner == (void*)fromPtr) {
                    *(void**)slot = (void*)toPtr;
                    modified = true;
                }
            }
            if (modified) {
                if (cudaGraphExecKernelNodeSetParams(exec, nd, &kp) == cudaSuccess)
                    ++patched;
            }
        }

        if (patched && c.debug.printHints) {
            std::fprintf(stderr,
                "[Graph][VelPatch] full=%d from=%p to=%p patched=%d\n",
                fullGraph ? 1 : 0,
                (const void*)fromPtr, (const void*)toPtr, patched);
        }
    }

    void Simulator::patchGraphHalfPositionPointers(bool fullGraph, sim::Half4* oldCurrH, sim::Half4* oldNextH) {
        const auto& c = console::Instance();
        if (!c.perf.graph_hot_update_enable) return;
        if (!m_bufs.nativeHalfActive) return; //仅原生 half 模式需要
        if (!oldCurrH || !oldNextH) return;
        cudaGraphExec_t exec = fullGraph ? m_graphExecFull : m_graphExecCheap;
        if (!exec) return;
        auto& nodes = fullGraph ? m_posNodesFull : m_posNodesCheap;
        if (nodes.empty()) return;
        int userLimit = c.perf.graph_hot_update_scan_limit;
        int scanLimit = (userLimit >0 && userLimit <=4096) ? userLimit :512;
        int patchedPtr =0; int patchedGrid =0;
        for (auto nd : nodes) {
            cudaKernelNodeParams kp{};
            if (cudaGraphKernelNodeGetParams(nd, &kp) != cudaSuccess) continue;
            if (!kp.kernelParams) continue;
            // 动态 gridDim 更新
            uint32_t blocks = (m_params.numParticles +255u) /256u; if (blocks ==0) blocks =1;
            if (kp.gridDim.x != blocks) { kp.gridDim.x = blocks; ++patchedGrid; }
            void** params = (void**)kp.kernelParams; bool modified = false;
            for (int i =0; i < scanLimit; ++i) {
                void* slot = params[i]; if (!slot) break; if (!hostPtrReadable(slot)) break;
                void* inner = *(void**)slot; if (!inner) continue;
                // 匹配旧 half 指针并替换
                if (inner == (void*)oldCurrH) { *(void**)slot = (void*)m_bufs.d_pos_h4; modified = true; }
                else if (inner == (void*)oldNextH) { *(void**)slot = (void*)m_bufs.d_pos_pred_h4; modified = true; }
            }
            if (modified || patchedGrid) {
                if (cudaGraphExecKernelNodeSetParams(exec, nd, &kp) == cudaSuccess) {
                    if (modified) ++patchedPtr;
                }
            }
        }
        if ((patchedPtr || patchedGrid) && c.debug.printHints) {
            std::fprintf(stderr,
                "[Graph][HalfPosPatch] full=%d patchedPtr=%d patchedGrid=%d currH=%p nextH=%p newCurrH=%p newNextH=%p\n",
                fullGraph ?1 :0, patchedPtr, patchedGrid,
                (void*)oldCurrH, (void*)oldNextH,
                (void*)m_bufs.d_pos_h4, (void*)m_bufs.d_pos_pred_h4);
        }
    }
    void Simulator::debugLogPredictedPosHalf(const SimParams& p) {
        const auto& cc = console::Instance();
        const auto& dbg = cc.debug;
        if (!dbg.logPredictedPosHalf) return;
        if (p.numParticles == 0) return;

        bool halfPred = (p.precision.predictedPosStore == NumericType::FP16 ||
            p.precision.predictedPosStore == NumericType::FP16_Packed);
        if (!halfPred) return;
        if (!m_bufs.nativeHalfActive && (!m_bufs.usePosPredHalf || !m_bufs.d_pos_pred_h4)) return;
        if (m_bufs.nativeHalfActive && !m_bufs.d_pos_pred_h4) return;

        int every = (dbg.logPredictedPosEveryN <= 0 ? 1 : dbg.logPredictedPosEveryN);
        if ((m_frameIndex % every) != 0) return;

        uint32_t N = p.numParticles;
        uint32_t stride = (dbg.logPredictedPosSampleStride == 0 ? 1u : (uint32_t)dbg.logPredictedPosSampleStride);
        uint32_t maxPrint = (dbg.logPredictedPosMaxPrint < 0 ? 0u : (uint32_t)dbg.logPredictedPosMaxPrint);

        // 抽样结构
        struct Sample {
            uint32_t i;
            float predHx, predHy, predHz; // half 解包预测
            float predFx, predFy, predFz; // float 预测（d_pos_pred）
            float currx, curry, currz;    // 当前
            float physDx, physDy, physDz; // float 预测 - 当前
            float quantDx, quantDy, quantDz; // half 解包 - float 预测
        };
        std::vector<Sample> samples;
        samples.reserve(maxPrint);

        // 指针属性验证（只打印一次，避免额外开销）
#if CUDART_VERSION >= 10000
        auto ptrDiag = [&](const char* tag, const void* ptr) {
            cudaPointerAttributes a{};
            if (cudaPointerGetAttributes(&a, ptr) != cudaSuccess) {
                if (dbg.printWarnings)
                    std::fprintf(stderr, "[PredPosHalf.Diag] %s getAttr failed ptr=%p\n", tag, ptr);
            }
            else {
                if (dbg.printDiagnostics)
                    std::fprintf(stderr, "[PredPosHalf.Diag] %s type=%d dev=%d ptr=%p\n",
                        tag, (int)a.type, a.device,  ptr);
            }
            };
        ptrDiag("curr", m_bufs.d_pos_curr);
        ptrDiag("pred_h4", m_bufs.d_pos_pred_h4);
        ptrDiag("pred_f", m_bufs.d_pos_pred);
#endif

        // 累计统计
        double sumPhysX = 0, sumPhysY = 0, sumPhysZ = 0;
        double sumPhysAbsX = 0, sumPhysAbsY = 0, sumPhysAbsZ = 0;
        double sumQuantX = 0, sumQuantY = 0, sumQuantZ = 0;
        double sumQuantAbsX = 0, sumQuantAbsY = 0, sumQuantAbsZ = 0;

        double sumPhysSqX = 0, sumPhysSqY = 0, sumPhysSqZ = 0;
        double sumQuantSqX = 0, sumQuantSqY = 0, sumQuantSqZ = 0;

        float minPhysX = FLT_MAX, minPhysY = FLT_MAX, minPhysZ = FLT_MAX;
        float maxPhysX = -FLT_MAX, maxPhysY = -FLT_MAX, maxPhysZ = -FLT_MAX;
        float minQuantX = FLT_MAX, minQuantY = FLT_MAX, minQuantZ = FLT_MAX;
        float maxQuantX = -FLT_MAX, maxQuantY = -FLT_MAX, maxQuantZ = -FLT_MAX;

        int abnormalCount = 0, nanCount = 0;
        int quantULP_GE1 = 0, quantULP_GE2 = 0, quantULP_GE4 = 0;
        int physStallCount = 0;          // 物理位移近似停滞
        int quantNonZeroWithPhysZero = 0;// 量化存在但物理位移近零
        int identicalHalfRepeat = 0;     // half 预测与前一个完全相同

        sim::Half4 hPred;
        float4 hCurr{}, hPredF{};
        bool haveFloatPred = (m_bufs.d_pos_pred != nullptr); // float 预测缓冲（alias next）
        sim::Half4 prevHPred{};
        bool prevValid = false;

        auto halfToF3 = [](const sim::Half4& h) {
            return make_float3(__half2float(h.x),
                __half2float(h.y),
                __half2float(h.z));
            };

        // half ULP 估算函数
        auto estimateHalfULP = [](float v)->float {
            float av = fabsf(v);
            if (av == 0.f) return __half2float((__half)1) * 0.f + 1.0f / 65536.0f;
            int exp2 = (int)floorf(log2f(av));
            // half mantissa bits =10 -> ULP=2^(exp-10)
            float ulp = powf(2.f, (float)(exp2 - 10));
            if (ulp < 1.0f / 65536.0f) ulp = 1.0f / 65536.0f;
            return ulp;
            };

        uint32_t visited = 0;
        for (uint32_t i = 0; i < N; i += stride) {
            ++visited;
            if (cudaMemcpy(&hPred, m_bufs.d_pos_pred_h4 + i, sizeof(sim::Half4), cudaMemcpyDeviceToHost) != cudaSuccess) break;
            if (m_bufs.d_pos_curr) {
                if (cudaMemcpy(&hCurr, m_bufs.d_pos_curr + i, sizeof(float4), cudaMemcpyDeviceToHost) != cudaSuccess) break;
            }
            else {
                hCurr = make_float4(0, 0, 0, 0);
            }
            if (haveFloatPred) {
                if (cudaMemcpy(&hPredF, m_bufs.d_pos_pred + i, sizeof(float4), cudaMemcpyDeviceToHost) != cudaSuccess) break;
            }
            else {
                hPredF = hCurr;
            }

            float3 predHalf3 = halfToF3(hPred);
            float physDx = hPredF.x - hCurr.x;
            float physDy = hPredF.y - hCurr.y;
            float physDz = hPredF.z - hCurr.z;

            float quantDx = predHalf3.x - hPredF.x;
            float quantDy = predHalf3.y - hPredF.y;
            float quantDz = predHalf3.z - hPredF.z;

            bool isNan = (!std::isfinite(predHalf3.x) || !std::isfinite(predHalf3.y) || !std::isfinite(predHalf3.z));
            if (isNan) ++nanCount;

            bool abnormal = (fabsf(predHalf3.x) > dbg.logPredictedPosAbsMax ||
                fabsf(predHalf3.y) > dbg.logPredictedPosAbsMax ||
                fabsf(predHalf3.z) > dbg.logPredictedPosAbsMax ||
                !std::isfinite(physDx) || !std::isfinite(physDy) || !std::isfinite(physDz) ||
                !std::isfinite(quantDx) || !std::isfinite(quantDy) || !std::isfinite(quantDz));
            if (abnormal) ++abnormalCount;

            // 统计
            sumPhysX += physDx; sumPhysY += physDy; sumPhysZ += physDz;
            sumPhysAbsX += fabs(physDx); sumPhysAbsY += fabs(physDy); sumPhysAbsZ += fabs(physDz);
            sumPhysSqX += physDx * physDx; sumPhysSqY += physDy * physDy; sumPhysSqZ += physDz * physDz;

            sumQuantX += quantDx; sumQuantY += quantDy; sumQuantZ += quantDz;
            sumQuantAbsX += fabs(quantDx); sumQuantAbsY += fabs(quantDy); sumQuantAbsZ += fabs(quantDz);
            sumQuantSqX += quantDx * quantDx; sumQuantSqY += quantDy * quantDy; sumQuantSqZ += quantDz * quantDz;

            minPhysX = fminf(minPhysX, physDx); maxPhysX = fmaxf(maxPhysX, physDx);
            minPhysY = fminf(minPhysY, physDy); maxPhysY = fmaxf(maxPhysY, physDy);
            minPhysZ = fminf(minPhysZ, physDz); maxPhysZ = fmaxf(maxPhysZ, physDz);

            minQuantX = fminf(minQuantX, quantDx); maxQuantX = fmaxf(maxQuantX, quantDx);
            minQuantY = fminf(minQuantY, quantDy); maxQuantY = fmaxf(maxQuantY, quantDy);
            minQuantZ = fminf(minQuantZ, quantDz); maxQuantZ = fmaxf(maxQuantZ, quantDz);

            // 颤动检测：物理位移极小但量化非零
            const float physEps = 1e-5f;
            bool physStall = (fabsf(physDx) < physEps && fabsf(physDy) < physEps && fabsf(physDz) < physEps);
            if (physStall) ++physStallCount;
            if (physStall && (fabsf(quantDx) > 0 || fabsf(quantDy) > 0 || fabsf(quantDz) > 0))
                ++quantNonZeroWithPhysZero;

            // 重复 half 预测
            if (prevValid &&
                hPred.x == prevHPred.x &&
                hPred.y == prevHPred.y &&
                hPred.z == prevHPred.z)
                ++identicalHalfRepeat;
            prevHPred = hPred; prevValid = true;

            // ULP 量化级别
            float ulpX = estimateHalfULP(hPredF.x);
            float ulpY = estimateHalfULP(hPredF.y);
            float ulpZ = estimateHalfULP(hPredF.z);
            auto countUlp = [&](float q, float ulp) {
                float aq = fabsf(q);
                if (aq >= ulp) ++quantULP_GE1;
                if (aq >= 2 * ulp) ++quantULP_GE2;
                if (aq >= 4 * ulp) ++quantULP_GE4;
                };
            countUlp(quantDx, ulpX);
            countUlp(quantDy, ulpY);
            countUlp(quantDz, ulpZ);

            if (samples.size() < maxPrint) {
                samples.push_back({ i,
                    predHalf3.x, predHalf3.y, predHalf3.z,
                    hPredF.x, hPredF.y, hPredF.z,
                    hCurr.x, hCurr.y, hCurr.z,
                    physDx, physDy, physDz,
                    quantDx, quantDy, quantDz });
            }
        }

        uint32_t sampleCount = visited;

        // 计算平均与 RMS
        auto avg = [&](double s) { return sampleCount ? (s / sampleCount) : 0.0; };
        auto rms = [&](double sq) { return sampleCount ? sqrt(sq / sampleCount) : 0.0; };

        double avgPhysX = avg(sumPhysX), avgPhysY = avg(sumPhysY), avgPhysZ = avg(sumPhysZ);
        double avgPhysAbsX = avg(sumPhysAbsX), avgPhysAbsY = avg(sumPhysAbsY), avgPhysAbsZ = avg(sumPhysAbsZ);
        double rmsPhysX = rms(sumPhysSqX), rmsPhysY = rms(sumPhysSqY), rmsPhysZ = rms(sumPhysSqZ);

        double avgQuantX = avg(sumQuantX), avgQuantY = avg(sumQuantY), avgQuantZ = avg(sumQuantZ);
        double avgQuantAbsX = avg(sumQuantAbsX), avgQuantAbsY = avg(sumQuantAbsY), avgQuantAbsZ = avg(sumQuantAbsZ);
        double rmsQuantX = rms(sumQuantSqX), rmsQuantY = rms(sumQuantSqY), rmsQuantZ = rms(sumQuantSqZ);

        // 比例
        double stallRatio = sampleCount ? (double)physStallCount / sampleCount : 0.0;
        double quantWhileStallRatio = sampleCount ? (double)quantNonZeroWithPhysZero / sampleCount : 0.0;
        double identicalRatio = sampleCount ? (double)identicalHalfRepeat / sampleCount : 0.0;

        if (dbg.logPredictedPosPrintHeader) {
            std::fprintf(stderr,
                "[PredPosHalf+][frame=%d] N=%u stride=%u samples=%u printed=%zu "
                "nan=%d abnormal=%d stall=%d(%.2f%%) q&stall=%d(%.2f%%) repeatHalf=%d(%.2f%%)\n"
                "  physΔ avg=(%.6e,%.6e,%.6e) avg|Δ|=(%.6e,%.6e,%.6e) rms=(%.6e,%.6e,%.6e) min=(%.6e,%.6e,%.6e) max=(%.6e,%.6e,%.6e)\n"
                "  quantΔ avg=(%.6e,%.6e,%.6e) avg|Δ|=(%.6e,%.6e,%.6e) rms=(%.6e,%.6e,%.6e) min=(%.6e,%.6e,%.6e) max=(%.6e,%.6e,%.6e)\n"
                "  quantULP>=1=%d quantULP>=2=%d quantULP>=4=%d storeType=%u native=%d curr=%p pred_f=%p pred_h4=%p\n",
                m_frameIndex, N, stride, sampleCount, samples.size(),
                nanCount, abnormalCount,
                physStallCount, stallRatio * 100.0,
                quantNonZeroWithPhysZero, quantWhileStallRatio * 100.0,
                identicalHalfRepeat, identicalRatio * 100.0,
                avgPhysX, avgPhysY, avgPhysZ,
                avgPhysAbsX, avgPhysAbsY, avgPhysAbsZ,
                rmsPhysX, rmsPhysY, rmsPhysZ,
                minPhysX, minPhysY, minPhysZ,
                maxPhysX, maxPhysY, maxPhysZ,
                avgQuantX, avgQuantY, avgQuantZ,
                avgQuantAbsX, avgQuantAbsY, avgQuantAbsZ,
                rmsQuantX, rmsQuantY, rmsQuantZ,
                minQuantX, minQuantY, minQuantZ,
                maxQuantX, maxQuantY, maxQuantZ,
                quantULP_GE1, quantULP_GE2, quantULP_GE4,
                (unsigned)p.precision.predictedPosStore,
                (int)m_bufs.nativeHalfActive,
                (void*)m_bufs.d_pos_curr,
                (void*)m_bufs.d_pos_pred,
                (void*)m_bufs.d_pos_pred_h4);
        }

        for (auto& s : samples) {
            std::fprintf(stderr,
                "[PredPosHalf.Sample] i=%u predH=(%.6f,%.6f,%.6f) predF=(%.6f,%.6f,%.6f) curr=(%.6f,%.6f,%.6f) "
                "physΔ=(%.6e,%.6e,%.6e) quantΔ=(%.6e,%.6e,%.6e)\n",
                s.i,
                s.predHx, s.predHy, s.predHz,
                s.predFx, s.predFy, s.predFz,
                s.currx, s.curry, s.currz,
                s.physDx, s.physDy, s.physDz,
                s.quantDx, s.quantDy, s.quantDz);
        }

        // 额外提示
        if (stallRatio > 0.5 && quantWhileStallRatio > 0.2 && cc.debug.printHints) {
            std::fprintf(stderr,
                "[PredPosHalf+][Hint] 超过 50%% 样本物理位移近零但量化误差存在 (%u/%u)。可能求解迭代未更新 float 预测或半精镜像过度重复 pack。\n",
                quantNonZeroWithPhysZero, sampleCount);
        }
        if (identicalRatio > 0.8 && cc.debug.printHints) {
            std::fprintf(stderr,
                "[PredPosHalf+][Hint] 超过 80%% 样本 half 预测与上一次完全一致，可能未刷新半精镜像（缺少 pack 触发）。\n");
        }
        if (nanCount > 0 && cc.debug.printErrors) {
            std::fprintf(stderr,
                "[PredPosHalf+][Error] 检测到 %d 个 NaN half 预测样本。检查外部缓冲初始化 / 写越界。\n", nanCount);
        }
        if (abnormalCount > 0 && cc.debug.printWarnings) {
            std::fprintf(stderr,
                "[PredPosHalf+][Warn] 异常样本=%d (%.2f%%)。阈值=%.3e。\n",
                abnormalCount, sampleCount ? (100.0 * abnormalCount / sampleCount) : 0.0, dbg.logPredictedPosAbsMax);
        }
    }
} // namespace sim