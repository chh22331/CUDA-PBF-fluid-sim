#pragma once
#include <cuda_runtime.h>
#include "parameters.h"
#include "device_buffers.cuh"
#include "grid_buffers.cuh"
#include "emit_params.h"
#include "emitter.h"
#include "logging.h"
#include "stats.h"
#include <vector>
#include "param_change_tracker.h"
#include "phase_pipeline.h"
#include "kernel_dispatcher.h"
#include "simulation_context.h"
#include "grid_strategy.h"
#include "post_ops.h"

namespace sim {

    struct OnlineVarHost { double mean = 0.0; double m2 = 0.0; uint64_t n = 0; void add(double x) { ++n; double d = x - mean; mean += d / double(n); double d2 = x - mean; m2 += d * d2; } double variance() const { return (n > 1) ? (m2 / double(n - 1)) : 0.0; } };

    struct PersistentKernelArgs {
        cudaGraphNode_t node{};
        const void* func = nullptr;
        cudaKernelNodeParams base{};      // 原始（用于gridDim等非参数值字段）
        std::vector<uint8_t> valueStorage; // 参数值副本（设备指针8字节；标量4字节）
        std::vector<void*>   paramPtrs;    // kernelParams -> 指向 valueStorage 各槽
        std::vector<uint8_t> isPtr;        // 1=设备/托管指针；0=标量
        uint32_t paramCount = 0;
    };

    class GraphBuilder;

    class Simulator {
    public:
        bool initialize(const SimParams& p);
        void shutdown();
        bool step(const SimParams& p);

        const float4* devicePositions() const { return m_bufs.d_pos_curr; }
        float4* renderPositionPtr() const { return m_bufs.d_pos_curr; }

        void seedBoxLattice(uint32_t nx, uint32_t ny, uint32_t nz, float3 origin, float spacing);
        void seedBoxLatticeAuto(uint32_t total, float3 origin, float spacing);
        uint32_t activeParticleCount() const { return m_params.numParticles; }
        bool computeStats(SimStats& out, uint32_t sampleStride = 4) const;
        bool computeStatsBruteforce(SimStats& out, uint32_t sampleStride, uint32_t maxISamples) const;
        bool computeCenterOfMass(float3& outCom, uint32_t sampleStride) const;
        void seedCubeMix(uint32_t groupCount, const float3* groupCenters, uint32_t edgeParticles, float spacing, bool applyJitter, float jitterAmp, uint32_t jitterSeed);
        bool adaptivePrecisionCheck(const SimStats& stats);
        void debugSampleDisplacement(uint32_t sampleStride = 1024);

        cudaStream_t cudaStream() const { return m_stream; }
        void syncForRender();

        bool bindTimelineFence(HANDLE sharedFenceHandle);
        uint64_t lastSimFenceValue() const { return m_simFenceValue; }

        // 新增：供 app_main.cpp 读取上一帧 GPU 侧模拟耗时（毫秒）
        double lastGpuFrameMs() const { return (double)m_lastFrameMs; }

        bool importRenderHalfBuffer(void* sharedHandleWin32, size_t bytes);
        void publishRenderHalf(uint32_t count);
        void releaseRenderHalfExternal();

    private:
        friend class GraphBuilder;

        bool buildGrid(const SimParams& p);
        bool ensureSortTemp(std::size_t bytes);
        bool structuralGraphChange(const SimParams& p) const;
        bool paramOnlyGraphChange(const SimParams& p) const;
        bool captureGraphIfNeeded(const SimParams& p);
        bool updateGraphsParams(const SimParams& p);
        bool updateGridIfNeeded(const SimParams& p);
        bool cacheGraphNodes();
        void patchGraphPositionPointers(float4* oldCurr, float4* oldNext);
        void patchGraphVelocityPointers(const float4* fromPtr, const float4* toPtr);
        void patchGraphHalfPositionPointers(sim::Half4* oldCurrH, sim::Half4* oldNextH);
        void signalSimFence();

    private:
        struct CachedKernelArgs {
            cudaGraphNode_t node{};
            cudaKernelNodeParams base{};
            std::vector<void*> argStorage;      // 每个参数值的“副本”存放处
            std::vector<void**> paramPtrs;      // kernelParams 用到的指针数组（持久）
            std::vector<uint8_t> isPtr;         // 1 表示此参数原本语义为指针（设备/主机）
        };

        std::vector<CachedKernelArgs> m_posCached;
        std::vector<CachedKernelArgs> m_velCached;

        SimParams m_params{};
        DeviceBuffers m_bufs{};
        GridBuffers   m_grid{};
        uint32_t m_numCells = 0;
        bool     m_useHashedGrid = false;
        uint32_t m_numCompactCells = 0;

        cudaStream_t m_stream = nullptr;
        cudaEvent_t  m_evStart[2] = { nullptr, nullptr };
        cudaEvent_t  m_evEnd[2]   = { nullptr, nullptr };
        int          m_evCursor   = 0;
        float        m_lastFrameMs = -1.0f; // 事件采样后填充，供 lastGpuFrameMs()
        int          m_frameTimingEveryN = 1;
        bool         m_frameTimingEnabled = true;

        cudaGraph_t     m_graphFull = nullptr;
        cudaGraphExec_t m_graphExecFull = nullptr;
        bool m_graphDirty = true;
        bool m_paramDirty = true;
        bool m_precisionLogged = false;
        bool m_graphPointersChecked = false;

        std::vector<cudaGraphNode_t> m_posNodesFull;
        std::vector<cudaKernelNodeParams> m_posNodeParamsBaseFull;
        bool m_cachedPosNodes = false;

        int  m_frameIndex = 0;
        int  m_lastParamUpdateFrame = -1;

        struct GraphCapturedParams {
            uint32_t numParticles = 0; uint32_t numCells = 0; int solverIters = 0; int maxNeighbors = 0;
            int sortEveryN = 1; GridBounds grid{}; KernelCoeffs kernel{}; float dt = 0.0f;
            float3 gravity = make_float3(0.f, 0.f, 0.f); float restDensity = 0.0f;
        } m_captured{};

        cudaExternalMemory_t m_extPosPred = nullptr;

        cudaGraphNode_t      m_nodeRecycleFull = nullptr;
        cudaKernelNodeParams m_kpRecycleBaseFull{};
        bool m_cachedNodesReady = false;

        std::vector<cudaGraphNode_t> m_velNodesFull;
        std::vector<cudaKernelNodeParams> m_velNodeParamsBaseFull;
        bool   m_cachedVelNodes = false;

        OnlineVarHost m_adaptDensityErrorHistory{};
        OnlineVarHost m_adaptLambdaVarHistory{};
        int  m_adaptUpgradeHold = 0;
        int  m_adaptDowngradeHold = 0;
        bool m_adaptHalfDisabled = false;

        ParamChangeTracker m_paramTracker;
        KernelDispatcher   m_kernelDispatcher;
        PhasePipeline      m_pipeline;
        SimulationContext  m_ctx;
        std::unique_ptr<IGridStrategy> m_gridStrategy;
        PostOpsPipeline    m_postPipeline;

        cudaExternalSemaphore_t m_extTimelineSem = nullptr;
        uint64_t m_simFenceValue = 0;

        cudaExternalMemory_t m_extRenderHalf = nullptr;
        void* m_renderHalfMappedPtr = nullptr;
        size_t m_renderHalfBytes = 0;

        std::vector<PersistentKernelArgs> m_persistentArgs;

        // ===== 重构后使用的缓存（取代 m_posCached/m_velCached 中 argStorage 的不稳定来源） =====
        std::vector<PersistentKernelArgs*> m_posNodesPersistent;
        std::vector<PersistentKernelArgs*> m_velNodesPersistent;

    };
} // namespace sim