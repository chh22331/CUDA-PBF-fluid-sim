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
#ifdef _WIN32
#include <windows.h>
#endif

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

namespace sim {

    uint64_t g_simFrameIndex = 0;

    // ===== 诊断枚举 =====
    enum class ParamSlotClass : uint8_t {
        NullSlot,
        DeviceValue,
        HostPtrReadable,
        HostPtrDangling,
        Misaligned,
        Unknown
    };

    struct ParamDiagRec {
        int index = -1;
        void* slot = nullptr;
        void* value = nullptr;
        ParamSlotClass cls = ParamSlotClass::Unknown;
        bool matchesPos = false;
        bool matchesVel = false;
        bool derefTried = false;
        bool derefSucceeded = false;
        cudaError_t attrErr = cudaSuccess;
        int attrType = -1;
    };

    // 保存每个节点原始参数布局（不再截断）
    struct PersistentKernelArgs {
        cudaGraphNode_t node = nullptr;
        void* func = nullptr;
        cudaKernelNodeParams base{}; // 原始参数结构
        uint32_t        paramCount = 0; // 原始槽数量（rawCount）
        // isPtr[i] 表示槽内容(解引用值)是否为设备/托管指针
        std::vector<uint8_t> isPtr;
        // 不再构造新 paramPtrs/valueStorage；保留以兼容旧逻辑，但为空
        std::vector<void*> paramPtrs;
        std::vector<uint8_t> valueStorage;
    };

    // ===== 诊断辅助函数 =====
    static bool IsReadableHostPointer(const void* p) {
#ifdef _WIN32
        if (!p) return false;
        MEMORY_BASIC_INFORMATION mbi{};
        if (VirtualQuery(p, &mbi, sizeof(mbi)) != sizeof(mbi)) return false;
        if (mbi.State != MEM_COMMIT) return false;
        if (mbi.Protect & PAGE_GUARD) return false;
        DWORD prot = mbi.Protect;
        const DWORD kReadable = PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY |
            PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
        return (prot & kReadable) != 0;
#else
        return p != nullptr;
#endif
    }

    static ParamDiagRec ClassifyKernelParamSlot(
        void* slot,
        const std::unordered_set<const void*>& watchPos,
        const std::unordered_set<const void*>& watchVel)
    {
        ParamDiagRec rec;
        rec.slot = slot;
        if (!slot) { rec.cls = ParamSlotClass::NullSlot; return rec; }
        if ((reinterpret_cast<uintptr_t>(slot) & (sizeof(void*) - 1)) != 0) {
            rec.cls = ParamSlotClass::Misaligned; return rec;
        }
        cudaPointerAttributes attr{};
        cudaError_t perr = cudaPointerGetAttributes(&attr, slot);
        rec.attrErr = perr;
#if CUDART_VERSION >= 10000
        if (perr == cudaSuccess &&
            (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged)) {
#else
        if (perr == cudaSuccess &&
            (attr.memoryType == cudaMemoryTypeDevice || attr.memoryType == cudaMemoryTypeManaged)) {
#endif
            rec.cls = ParamSlotClass::DeviceValue;
            rec.value = slot;
            rec.matchesPos = watchPos.count(rec.value) != 0;
            rec.matchesVel = watchVel.count(rec.value) != 0;
            return rec;
        }
        if (!IsReadableHostPointer(slot)) {
            rec.cls = ParamSlotClass::HostPtrDangling;
            return rec;
        }
        rec.derefTried = true;
        void* val = *(void**)slot;
        rec.value = val;
        rec.derefSucceeded = true;
        rec.cls = ParamSlotClass::HostPtrReadable;
        if (val) {
            rec.matchesPos = watchPos.count(val) != 0;
            rec.matchesVel = watchVel.count(val) != 0;
        }
        return rec;
        }

    static const char* SlotClassStr(ParamSlotClass c) {
        switch (c) {
        case ParamSlotClass::NullSlot: return "Null";
        case ParamSlotClass::DeviceValue: return "DeviceValue";
        case ParamSlotClass::HostPtrReadable: return "HostPtrReadable";
        case ParamSlotClass::HostPtrDangling: return "HostPtrDangling";
        case ParamSlotClass::Misaligned: return "Misaligned";
        default: return "Unknown";
        }
    }

    static bool EnableGraphParamDiag() {
        // 使用 printWarnings 控制；也可加环境变量强制开启
        const auto& cc = console::Instance();
        static bool envForce = (std::getenv("SIM_GRAPH_DIAG_FORCE") != nullptr);
        return cc.debug.printWarnings || envForce;
    }

    static inline bool SafeParamSlot(void** base, int idx, int maxSlots) {
        return base && idx >= 0 && idx < maxSlots;
    }

    static inline bool TryReadArgPtr(void** slot, void*& outValue) {
        if (!slot) return false;
        if ((reinterpret_cast<uintptr_t>(slot) & (sizeof(void*) - 1)) != 0) return false;
        outValue = *slot;
        return true;
    }

    // ===== 外部共享半精缓冲 =====
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
        cudaExternalMemoryBufferDesc bufDesc{};
        bufDesc.offset = 0;
        bufDesc.size = bytes;
        void* devPtr = nullptr;
        if (cudaExternalMemoryGetMappedBuffer(&devPtr, m_extRenderHalf, &bufDesc) != cudaSuccess) {
            std::fprintf(stderr, "[RenderHalf][Map] failed\n");
            cudaDestroyExternalMemory(m_extRenderHalf);
            m_extRenderHalf = nullptr;
            return false;
        }
        m_renderHalfMappedPtr = devPtr;
        m_renderHalfBytes = bytes;
        std::fprintf(stderr, "[RenderHalf][Ready] mappedPtr=%p bytes=%zu\n",
            m_renderHalfMappedPtr, m_renderHalfBytes);
        return true;
    }

    void Simulator::publishRenderHalf(uint32_t count) {
        if (!m_renderHalfMappedPtr || !m_extRenderHalf || count == 0) return;
        if (!m_bufs.useRenderHalf || !m_bufs.d_render_pos_h4) return;
        size_t needBytes = size_t(count) * sizeof(sim::Half4);
        if (needBytes > m_renderHalfBytes) {
            std::fprintf(stderr, "[RenderHalf][Warn] external buffer too small need=%zu have=%zu\n",
                needBytes, m_renderHalfBytes);
            return;
        }
        if (m_bufs.nativeHalfActive) {
            CUDA_CHECK(cudaMemcpyAsync(m_renderHalfMappedPtr, m_bufs.d_pos_h4, needBytes,
                cudaMemcpyDeviceToDevice, m_stream));
        }
        else {
            m_bufs.packRenderToHalf(count, m_stream);
            CUDA_CHECK(cudaMemcpyAsync(m_renderHalfMappedPtr, m_bufs.d_render_pos_h4, needBytes,
                cudaMemcpyDeviceToDevice, m_stream));
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

    // ===== 原位 Patch 辅助：在原始 kernelParams 数组中修改指针值 =====
    static bool InplaceReplacePtr(void** slotPtr, void* oldVal, void* newVal) {
        if (!slotPtr || !*slotPtr) return false;
        void* cur = nullptr;
        std::memcpy(&cur, *slotPtr, sizeof(void*)); // *slotPtr 指向参数值内存
        if (cur == oldVal && newVal) {
            std::memcpy(*slotPtr, &newVal, sizeof(void*));
            return true;
        }
        return false;
    }

    void Simulator::patchGraphHalfPositionPointers(sim::Half4* oldCurrH, sim::Half4* oldNextH) {
        if (!m_graphExecFull || !m_bufs.nativeHalfActive || !oldCurrH || !oldNextH) return;
        int nodePatched = 0, ptrPatched = 0;
        for (auto* pk : m_posNodesPersistent) {
            if (!pk || !pk->base.kernelParams) continue;
            void** slots = (void**)pk->base.kernelParams;
            bool changed = false;
            for (uint32_t i = 0; i < pk->paramCount; ++i) {
                if (!SafeParamSlot(slots, (int)i, (int)pk->paramCount)) break;
                if (InplaceReplacePtr(&slots[i], (void*)oldCurrH, (void*)m_bufs.d_pos_h4)) { changed = true; ++ptrPatched; }
                if (InplaceReplacePtr(&slots[i], (void*)oldNextH, (void*)m_bufs.d_pos_pred_h4)) { changed = true; ++ptrPatched; }
            }
            if (changed) {
                ++nodePatched;
                if (console::Instance().debug.printHints) {
                    std::fprintf(stderr, "[HalfPosPatchInPlace] node=%p currHNew=%p nextHNew=%p paramCount=%u\n",
                        (void*)pk->node, (void*)m_bufs.d_pos_h4, (void*)m_bufs.d_pos_pred_h4, pk->paramCount);
                }
            }
        }
        if (console::Instance().debug.printWarnings && (nodePatched || ptrPatched))
            std::fprintf(stderr, "[HalfPosPatchInPlace][Summary] nodes=%d ptrs=%d\n", nodePatched, ptrPatched);
    }

    void Simulator::patchGraphVelocityPointers(const float4 * fromPtr, const float4 * toPtr) {
        if (!m_graphExecFull || !fromPtr || !toPtr || fromPtr == toPtr) return;
        int nodePatched = 0;
        int ptrPatched = 0;
        for (auto* pk : m_velNodesPersistent) {
            if (!pk || !pk->base.kernelParams) continue;
            void** slots = (void**)pk->base.kernelParams;
            bool changed = false;
            for (uint32_t i = 0; i < pk->paramCount; ++i) {
                if (!SafeParamSlot(slots, i, (int)pk->paramCount)) break;
                if (InplaceReplacePtr(&slots[i], (void*)fromPtr, (void*)toPtr)) {
                    changed = true; ++ptrPatched;
                }
            }
            if (changed) {
                ++nodePatched;
                if (console::Instance().debug.printHints) {
                    std::fprintf(stderr,
                        "[VelPatchInPlace] node=%p from=%p to=%p paramCount=%u\n",
                        (void*)pk->node, (const void*)fromPtr, (const void*)toPtr, pk->paramCount);
                }
            }
        }
        if (console::Instance().debug.printWarnings && (nodePatched || ptrPatched)) {
            std::fprintf(stderr, "[VelPatchInPlace][Summary] nodes=%d ptrs=%d\n", nodePatched, ptrPatched);
        }
    }

    void Simulator::patchGraphHalfPositionPointers(sim::Half4 * oldCurrH, sim::Half4 * oldNextH) {
        if (!m_graphExecFull || !m_bufs.nativeHalfActive || !oldCurrH || !oldNextH) return;
        int nodePatched = 0;
        int ptrPatched = 0;
        for (auto* pk : m_posNodesPersistent) {
            if (!pk || !pk->base.kernelParams) continue;
            void** slots = (void**)pk->base.kernelParams;
            bool changed = false;
            for (uint32_t i = 0; i < pk->paramCount; ++i) {
                if (!SafeParamSlot(slots, i, (int)pk->paramCount)) break;
                if (InplaceReplacePtr(&slots[i], (void*)oldCurrH, (void*)m_bufs.d_pos_h4)) { changed = true; ++ptrPatched; }
                if (InplaceReplacePtr(&slots[i], (void*)oldNextH, (void*)m_bufs.d_pos_pred_h4)) { changed = true; ++ptrPatched; }
            }
            if (changed) {
                ++nodePatched;
                if (console::Instance().debug.printHints) {
                    std::fprintf(stderr,
                        "[HalfPosPatchInPlace] node=%p currHNew=%p nextHNew=%p paramCount=%u\n",
                        (void*)pk->node, (void*)m_bufs.d_pos_h4, (void*)m_bufs.d_pos_pred_h4, pk->paramCount);
                }
            }
        }
        if (console::Instance().debug.printWarnings && (nodePatched || ptrPatched)) {
            std::fprintf(stderr, "[HalfPosPatchInPlace][Summary] nodes=%d ptrs=%d\n", nodePatched, ptrPatched);
        }
    }

    // ===== 初始化 =====
    bool Simulator::initialize(const SimParams & p) {
        prof::Range rInit("Sim.Initialize", prof::Color(0x10, 0x90, 0xF0));
        m_params = p;
        CUDA_CHECK(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));

        const auto& c = console::Instance();
        m_frameTimingEveryN = c.perf.frame_timing_every_n;
        m_frameTimingEnabled = (m_frameTimingEveryN != 0);
        m_useHashedGrid = c.perf.use_hashed_grid;

        m_ctx.bufs = &m_bufs;
        m_ctx.grid = &m_grid;
        m_ctx.useHashedGrid = m_useHashedGrid;
        if (m_useHashedGrid) m_gridStrategy = std::make_unique<HashedGridStrategy>();
        else m_gridStrategy = std::make_unique<DenseGridStrategy>();
        m_ctx.gridStrategy = m_gridStrategy.get();
        m_ctx.dispatcher = &m_kernelDispatcher;

        uint32_t capacity = (p.maxParticles > 0) ? p.maxParticles : p.numParticles;
        if (capacity == 0) capacity = 1;

        bool needHalf =
            (p.precision.positionStore == NumericType::FP16_Packed) ||
            (p.precision.velocityStore == NumericType::FP16_Packed) ||
            (p.precision.predictedPosStore == NumericType::FP16_Packed) ||
            (p.precision.lambdaStore == NumericType::FP16) ||
            (p.precision.densityStore == NumericType::FP16) ||
            (p.precision.auxStore == NumericType::FP16_Packed || p.precision.auxStore == NumericType::FP16);

        if (needHalf) m_bufs.allocateWithPrecision(p.precision, capacity);
        else m_bufs.allocate(capacity);

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
            CUDA_CHECK(cudaMemcpy(m_bufs.d_pos_next, m_bufs.d_pos,
                sizeof(float4) * p.numParticles, cudaMemcpyDeviceToDevice));
            m_bufs.d_pos_pred = m_bufs.d_pos_next;
            if (needHalf) m_bufs.packAllToHalf(p.numParticles, m_stream);
        }
        UploadSimPosTableConst(m_bufs.d_pos_curr, m_bufs.d_pos_next);

        m_graphDirty = true;
        m_captured = {};
        m_cachedNodesReady = false;
        m_nodeRecycleFull = nullptr;
        m_lastFrameMs = -1.0f;
        m_evCursor = 0;
        m_lastParamUpdateFrame = -1;

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
            "[Init][FullGraphOnly] curr=%p next=%p pred(alias next)=%p N=%u\n",
            (void*)m_bufs.d_pos_curr, (void*)m_bufs.d_pos_next,
            (void*)m_bufs.d_pos_pred, p.numParticles);

        return true;
    }

    void Simulator::shutdown() {
        if (m_extPosPred) {
            cudaDestroyExternalMemory(m_extPosPred);
            m_extPosPred = nullptr;
        }
        if (m_extRenderHalf) {
            cudaDestroyExternalMemory(m_extRenderHalf);
            m_extRenderHalf = nullptr;
        }
        m_grid.releaseAll();
        if (m_graphExecFull) { cudaGraphExecDestroy(m_graphExecFull); m_graphExecFull = nullptr; }
        if (m_graphFull) { cudaGraphDestroy(m_graphFull); m_graphFull = nullptr; }

        for (int i = 0; i < 2; ++i) {
            if (m_evStart[i]) { cudaEventDestroy(m_evStart[i]); m_evStart[i] = nullptr; }
            if (m_evEnd[i]) { cudaEventDestroy(m_evEnd[i]); m_evEnd[i] = nullptr; }
        }
        if (m_stream) { cudaStreamDestroy(m_stream); m_stream = nullptr; }
    }

    bool Simulator::buildGrid(const SimParams & p) {
        int3 dim = GridSystem::ComputeDims(p.grid);
        m_numCells = GridSystem::NumCells(dim);
        if (m_numCells == 0) return false;
        m_grid.allocGridRanges(m_numCells);
        m_params.grid = p.grid;
        m_params.grid.dim = dim;
        return true;
    }

    bool Simulator::updateGridIfNeeded(const SimParams & p) {
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

    // ===== 持久化节点缓存（原始参数布局诊断） =====
    bool Simulator::cacheGraphNodes() {
        m_nodeRecycleFull = nullptr;
        m_kpRecycleBaseFull = {};
        m_cachedNodesReady = false;

        m_velNodesFull.clear();
        m_velNodeParamsBaseFull.clear();
        m_posNodesFull.clear();
        m_posNodeParamsBaseFull.clear();
        m_cachedVelNodes = false;

        m_persistentArgs.clear();
        m_posNodesPersistent.clear();
        m_velNodesPersistent.clear();

        void* target = GetRecycleKernelPtr();
        if (!m_graphFull) {
            m_cachedNodesReady = true; return true;
        }

        size_t n = 0;
        CUDA_CHECK(cudaGraphGetNodes(m_graphFull, nullptr, &n));
        if (!n) {
            m_cachedNodesReady = true; return true;
        }
        std::vector<cudaGraphNode_t> nodes(n);
        CUDA_CHECK(cudaGraphGetNodes(m_graphFull, nodes.data(), &n));

        std::unordered_set<const void*> watchVel{
         (const void*)m_bufs.d_vel,
         (const void*)m_bufs.d_delta
        };
        std::unordered_set<const void*> watchPos;
        if (m_bufs.nativeHalfActive) {
            if (m_bufs.d_pos_h4) watchPos.insert((const void*)m_bufs.d_pos_h4);
            if (m_bufs.d_pos_pred_h4) watchPos.insert((const void*)m_bufs.d_pos_pred_h4);
        }
        else {
            if (m_bufs.d_pos_curr) watchPos.insert((const void*)m_bufs.d_pos_curr);
            if (m_bufs.d_pos_next) watchPos.insert((const void*)m_bufs.d_pos_next);
            if (m_bufs.d_pos)      watchPos.insert((const void*)m_bufs.d_pos);
            if (m_bufs.d_pos_pred) watchPos.insert((const void*)m_bufs.d_pos_pred);
        }

        bool diag = EnableGraphParamDiag();
        int printed = 0;
        const int printLimit = 512;

        for (auto nd : nodes) {
            cudaGraphNodeType t;
            CUDA_CHECK(cudaGraphNodeGetType(nd, &t));
            if (t != cudaGraphNodeTypeKernel) continue;

            cudaKernelNodeParams kp{};
            CUDA_CHECK(cudaGraphKernelNodeGetParams(nd, &kp));

            if (kp.func == target) {
                m_nodeRecycleFull = nd;
                m_kpRecycleBaseFull = kp;
            }

            if (!kp.kernelParams) continue;
            void** params = (void**)kp.kernelParams;

            // 扫描真实槽数量（直到遇到 nullptr 或非对齐）
            int rawCount = 0;
            for (int i = 0; i < 128; ++i) {
                void* slot = params[i];
                if (!slot) break;
                if ((reinterpret_cast<uintptr_t>(slot) & (sizeof(void*) - 1)) != 0) break;
                rawCount++;
            }

            // 记录
            PersistentKernelArgs pk;
            pk.node = nd;
            pk.func = kp.func;
            pk.base = kp;
            pk.paramCount = (uint32_t)rawCount;
            pk.isPtr.assign(rawCount, 0);

            bool hasPos = false, hasVel = false;

            for (int i = 0; i < rawCount; ++i) {
                void* slotPtr = params[i];
                if (!slotPtr) break;
                // 解引用槽值（真实参数值地址内容 8字节)
                void* val = nullptr;
                std::memcpy(&val, slotPtr, sizeof(void*));
                cudaPointerAttributes attr{};
                cudaError_t perr = cudaPointerGetAttributes(&attr, val);
#if CUDART_VERSION >= 10000
                bool isDev = (perr == cudaSuccess &&
                    (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged));
#else
                bool isDev = (perr == cudaSuccess &&
                    (attr.memoryType == cudaMemoryTypeDevice || attr.memoryType == cudaMemoryTypeManaged));
#endif
                pk.isPtr[i] = isDev ? 1 : 0;
                if (watchPos.count(val)) hasPos = true;
                if (watchVel.count(val)) hasVel = true;

                if (diag && printed < printLimit) {
                    unsigned long long raw64 = 0ULL;
                    std::memcpy(&raw64, slotPtr, sizeof(unsigned long long));
                    std::fprintf(stderr,
                        "[OrigParam] node=%p idx=%02d slotPtr=%p val=%p raw64=0x%016llX devPtr=%d posMatch=%d velMatch=%d perr=%d\n",
                        (void*)nd, i, slotPtr, val, raw64, (int)isDev,
                        (int)(watchPos.count(val) != 0), (int)(watchVel.count(val) != 0), (int)perr);
                }
            }

            m_persistentArgs.push_back(std::move(pk));
            PersistentKernelArgs* pkPtr = &m_persistentArgs.back();
            if (hasPos) m_posNodesPersistent.push_back(pkPtr);
            if (hasVel) m_velNodesPersistent.push_back(pkPtr);

            if (diag && printed < printLimit) {
                std::fprintf(stderr,
                    "[NodeSummary] node=%p func=%p rawCount=%d hasPos=%d hasVel=%d kernelParams=%p\n",
                    (void*)nd, kp.func, rawCount, (int)hasPos, (int)hasVel, kp.kernelParams);
                ++printed;
            }
        }

        if (diag) {
            std::fprintf(stderr,
                "[GraphDiag][Summary] persistent=%zu posPersist=%zu velPersist=%zu recycleNode=%p exec=%p\n",
                m_persistentArgs.size(), m_posNodesPersistent.size(),
                m_velNodesPersistent.size(), (void*)m_nodeRecycleFull, (void*)m_graphExecFull);
        }

        m_cachedVelNodes = true;
        m_cachedPosNodes = true;
        m_cachedNodesReady = true;
        return true;
    }

    // ===== Graph 变化判定 =====
    bool Simulator::structuralGraphChange(const SimParams & p) const {
        if (!m_graphExecFull) return true;
        if (p.solverIters != m_captured.solverIters) return true;
        if (p.maxNeighbors != m_captured.maxNeighbors) return true;
        if (p.numParticles != m_captured.numParticles) return true;
        int3 dim = GridSystem::ComputeDims(p.grid);
        uint32_t nc = GridSystem::NumCells(dim);
        if (nc != m_captured.numCells) return true;
        if (!gridEqual(p.grid, m_captured.grid)) return true;
        return false;
    }

    bool Simulator::paramOnlyGraphChange(const SimParams & p) const {
        if (structuralGraphChange(p)) return false;
        float dtRel = fabsf(p.dt - m_captured.dt) /
            fmaxf(1e-9f, fmaxf(p.dt, m_captured.dt));
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

    // ===== Graph 捕获 / 更新 =====
    bool Simulator::captureGraphIfNeeded(const SimParams & p) {
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

    bool Simulator::updateGraphsParams(const SimParams & p) {
        if (!m_paramDirty) return true;
        if (!m_graphExecFull) return false;
        prof::Range r("Sim.GraphUpdate", prof::Color(0xA0, 0x50, 0x10));
        GraphBuilder builder;
        auto res = builder.UpdateDynamic(*this, p, 0, m_frameIndex, m_lastParamUpdateFrame);
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

    // ===== 单步推进 =====
    bool Simulator::step(const SimParams & p) {
        prof::Range rf("Sim.Step", prof::Color(0x30, 0x30, 0xA0));
        g_simFrameIndex = m_frameIndex;
        m_params = p;

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
            bool needHalf2 =
                (pr.positionStore == NumericType::FP16_Packed ||
                    pr.velocityStore == NumericType::FP16_Packed ||
                    pr.lambdaStore == NumericType::FP16 ||
                    pr.densityStore == NumericType::FP16 ||
                    pr.auxStore == NumericType::FP16_Packed || pr.auxStore == NumericType::FP16);
            float4* oldCurr = m_bufs.d_pos_curr;
            float4* oldNext = m_bufs.d_pos_next;
            sim::Half4* oldCurrH = m_bufs.d_pos_h4;
            sim::Half4* oldNextH = m_bufs.d_pos_pred_h4;

            if (needHalf2) m_bufs.allocateWithPrecision(pr, m_params.numParticles);
            else m_bufs.allocate(m_params.numParticles);

            UpdateDevicePrecisionView(m_bufs, m_params.precision);

            m_grid.allocateIndices(m_bufs.capacity);
            std::vector<uint32_t> h_idx(m_bufs.capacity);
            for (uint32_t i = 0; i < m_bufs.capacity; ++i) h_idx[i] = i;
            CUDA_CHECK(cudaMemcpy(m_grid.d_indices, h_idx.data(),
                sizeof(uint32_t) * m_bufs.capacity, cudaMemcpyHostToDevice));
            m_graphDirty = true;
            UploadSimPosTableConst(m_bufs.d_pos_curr, m_bufs.d_pos_next);

            if (m_graphExecFull) {
                patchGraphPositionPointers(oldCurr, oldNext);
                if (m_bufs.nativeHalfActive) {
                    patchGraphHalfPositionPointers(oldCurrH, oldNextH);
                }
            }
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
                if (!m_evEnd[i]) cudaEventCreate(&m_evEnd[i]);
            }
            if (m_evStart[cur] && m_evEnd[cur])
                CUDA_CHECK(cudaEventRecord(m_evStart[cur], m_stream));
        }

        if (!m_graphExecFull) {
            if (console::Instance().debug.printErrors)
                std::fprintf(stderr, "[Step][Error] Graph not ready.\n");
            return false;
        }

        auto updNode = [&](cudaGraphExec_t exec,
            cudaGraphNode_t& node,
            cudaKernelNodeParams& base) {
                if (!exec || !node || m_params.numParticles == 0) return;
                uint32_t blocks = (m_params.numParticles + 255u) / 256u;
                if (blocks == 0) blocks = 1;
                cudaKernelNodeParams kp = base;
                kp.gridDim = dim3(blocks, 1, 1);
                // 修改 gridDim 不改 kernelParams 指针
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
        updNode(m_graphExecFull, m_nodeRecycleFull, m_kpRecycleBaseFull);

        CUDA_CHECK(cudaGraphLaunch(m_graphExecFull, m_stream));

        signalSimFence();

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
        ++m_evCursor;
        ++m_frameIndex;

        if (p.precision.renderTransfer == NumericType::FP16_Packed ||
            p.precision.renderTransfer == NumericType::FP16) {
            publishRenderHalf(m_params.numParticles);
        }
        return true;
    }

    // ===== 种子函数 =====
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

        auto cap = [&spacing](float len) -> uint32_t {
            return (spacing > 0.f) ? (uint32_t)floorf(len / spacing) : 0u;
            };

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
            if (nx < maxX) ++nx;
            else if (ny < maxY) ++ny;
            else if (nz < maxZ) ++nz;
            else break;
        }

        uint64_t nreq64 = (uint64_t)nx * ny * nz;
        uint32_t Nreq = (nreq64 > UINT32_MAX) ? UINT32_MAX : (uint32_t)nreq64;
        if (Nreq < total) {
            const auto& c = console::Instance();
            if (c.debug.printWarnings)
                std::fprintf(stderr, "[Warn] seedBoxLatticeAuto: Nreq(%u) < total(%u)\n", Nreq, total);
        }

        float3 margin = make_float3(spacing * 0.25f, spacing * 0.25f, spacing * 0.25f);
        origin.x = fminf(fmaxf(origin.x, m_params.grid.mins.x + margin.x), m_params.grid.maxs.x - margin.x);
        origin.y = fminf(fmaxf(origin.y, m_params.grid.mins.y + margin.y), m_params.grid.maxs.y - margin.y);
        origin.z = fminf(fmaxf(origin.z, m_params.grid.mins.z + margin.z), m_params.grid.maxs.z - margin.z);

        seedBoxLattice(nx, ny, nz, origin, spacing);
    }

    void Simulator::seedCubeMix(uint32_t groupCount, const float3 * centers, uint32_t edgeParticles,
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

    bool Simulator::computeStats(SimStats & out, uint32_t sampleStride) const {
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
        out.avgRhoRel = (m_params.restDensity > 0.f) ?
            (avgR_g / (double)m_params.restDensity) : 0.0;
        return true;
    }

    bool Simulator::computeStatsBruteforce(SimStats & out, uint32_t sampleStride,
        uint32_t maxISamples) const {
        if (m_params.numParticles == 0) { out = {}; return true; }
        double avgN = 0.0, avgV = 0.0, avgRhoRel = 0.0, avgR = 0.0;
        bool ok = LaunchComputeStatsBruteforce(m_bufs.d_pos_next, m_bufs.d_vel,
            m_params.kernel, m_params.particleMass,
            m_params.numParticles,
            (sampleStride == 0 ? 1u : sampleStride),
            maxISamples, &avgN, &avgV, &avgRhoRel, &avgR, m_stream);
        if (!ok) return false;
        out.N = m_params.numParticles;
        out.avgNeighbors = avgN;
        out.avgSpeed = avgV;
        out.avgRho = avgR;
        out.avgRhoRel = (m_params.restDensity > 0.f) ?
            (avgR / (double)m_params.restDensity) : 0.0;
        return true;
    }

    bool Simulator::computeCenterOfMass(float3 & outCom, uint32_t sampleStride) const {
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

    void Simulator::syncForRender() {
        if (m_stream) {
            cudaStreamSynchronize(m_stream);
        }
    }

    bool Simulator::bindTimelineFence(HANDLE sharedFenceHandle) {
        if (!sharedFenceHandle) return false;
        cudaExternalSemaphoreHandleDesc desc{};
        desc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
        desc.handle.win32.handle = sharedFenceHandle;
        desc.flags = 0;
        if (cudaImportExternalSemaphore(&m_extTimelineSem, &desc) != cudaSuccess) {
            std::fprintf(stderr, "[Sim][Timeline] import external fence failed\n");
            return false;
        }
        m_simFenceValue = 0;
        return true;
    }

    void Simulator::signalSimFence() {
        if (!m_extTimelineSem) return;
        ++m_simFenceValue;
        cudaExternalSemaphoreSignalParams params{};
        params.params.fence.value = m_simFenceValue;
        params.flags = 0;
        cudaSignalExternalSemaphoresAsync(&m_extTimelineSem, &params, 1, m_stream);
    }

    void Simulator::debugSampleDisplacement(uint32_t /*sampleStride*/) {
    }

    } // namespace sim