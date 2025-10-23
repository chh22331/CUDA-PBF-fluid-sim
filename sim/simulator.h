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

    struct OnlineVarHost { double mean=0.0; double m2=0.0; uint64_t n=0; void add(double x){ ++n; double d=x-mean; mean+=d/double(n); double d2=x-mean; m2+=d*d2; } double variance() const { return (n>1)?(m2/double(n-1)):0.0; } };

    class GraphBuilder;

    class Simulator {
    public:
        bool initialize(const SimParams& p);
        void shutdown();
        bool step(const SimParams& p);

        const float4* devicePositions() const { return m_bufs.d_pos_curr; }
        float4* renderPositionPtr() const { return m_bufs.d_pos_curr; } // 渲染使用 curr

        void seedBoxLattice(uint32_t nx, uint32_t ny, uint32_t nz, float3 origin, float spacing);
        void seedBoxLatticeAuto(uint32_t total, float3 origin, float spacing);
        bool importPosPredFromD3D12(void* sharedHandleWin32, size_t bytes);
        bool bindExternalPosPingPong(void* sharedHandleA, size_t bytesA, void* sharedHandleB, size_t bytesB);
        uint32_t activeParticleCount() const { return m_params.numParticles; }
        bool computeStats(SimStats& out, uint32_t sampleStride = 4) const;
        bool computeStatsBruteforce(SimStats& out, uint32_t sampleStride, uint32_t maxISamples) const;
        bool computeCenterOfMass(float3& outCom, uint32_t sampleStride) const;
        double lastGpuFrameMs() const { return static_cast<double>(m_lastFrameMs); }
        void seedCubeMix(uint32_t groupCount, const float3* groupCenters, uint32_t edgeParticles, float spacing, bool applyJitter, float jitterAmp, uint32_t jitterSeed);
        bool adaptivePrecisionCheck(const SimStats& stats);
        float4* pingpongPosA() const { return (m_bufs.externalPingPong ? m_bufs.d_pos_curr : nullptr); }
        float4* pingpongPosB() const { return (m_bufs.externalPingPong ? m_bufs.d_pos_next : nullptr); }
        bool externalPingPongEnabled() const { return m_bufs.externalPingPong; }
        void debugSampleDisplacement(uint32_t sampleStride = 1024);

        cudaStream_t cudaStream() const { return m_stream; }
        void syncForRender(); // 保留
        bool swappedThisFrame() const { return m_swappedThisFrame; }

        // 时间线 fence 绑定（D3D12 shared fence -> CUDA external semaphore）
        bool bindTimelineFence(HANDLE sharedFenceHandle);
        uint64_t lastSimFenceValue() const { return m_simFenceValue; } // 最近一次模拟完成的奇数值

        // 半精渲染共享：导入 D3D12 half 压缩位置缓冲，并在每帧发布
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
        void kIntegratePred(cudaStream_t s, const SimParams& p);
        void kHashKeys(cudaStream_t s, const SimParams& p);
        void kSort(cudaStream_t s, const SimParams& p);
        void kCellRanges(cudaStream_t s, const SimParams& p);
        void kCellRangesCompact(cudaStream_t s, const SimParams& p);
        void kSolveIter(cudaStream_t s, const SimParams& p);
        void kVelocityAndPost(cudaStream_t s, const SimParams& p);
        bool cacheGraphNodes();
        void patchGraphPositionPointers(bool fullGraph,float4* oldCurr,float4* oldNext,float4* oldPred);        
        void patchGraphVelocityPointers(bool fullGraph, const float4* fromPtr, const float4* toPtr);
        void signalSimFence(); // 末尾 signal external semaphore

    private:
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
        float        m_lastFrameMs = -1.0f;
        int          m_frameTimingEveryN = 1;
        bool         m_frameTimingEnabled = true;

        bool         m_swappedThisFrame = false; // ping-pong swap indicator

        cudaGraph_t     m_graphFull = nullptr;
        cudaGraphExec_t m_graphExecFull = nullptr;
        cudaGraph_t     m_graphCheap = nullptr;
        cudaGraphExec_t m_graphExecCheap = nullptr;
        bool m_graphDirty = true;
        bool m_paramDirty = true;
        bool m_canPingPongPos = true;
        bool m_precisionLogged = false;
        bool m_graphPointersChecked = false;
        bool m_graphNodesPatchedOnce = false;

        std::vector<cudaGraphNode_t> m_posNodesFull,  m_posNodesCheap;
        std::vector<cudaKernelNodeParams> m_posNodeParamsBaseFull, m_posNodeParamsBaseCheap;
        bool m_cachedPosNodes = false;

        int  m_frameIndex = 0;
        int  m_lastFullFrame = -1;
        int  m_lastParamUpdateFrame = -1;

        struct GraphCapturedParams { uint32_t numParticles=0; uint32_t numCells=0; int solverIters=0; int maxNeighbors=0; int sortEveryN=1; GridBounds grid{}; KernelCoeffs kernel{}; float dt=0.0f; float3 gravity=make_float3(0.f,0.f,0.f); float restDensity=0.0f; } m_captured{};

        cudaExternalMemory_t m_extPosPred = nullptr;
        cudaExternalMemory_t m_extraExternalMemB = nullptr;

        cudaGraphNode_t      m_nodeRecycleFull  = nullptr;
        cudaGraphNode_t      m_nodeRecycleCheap = nullptr;
        cudaKernelNodeParams m_kpRecycleBaseFull{};
        cudaKernelNodeParams m_kpRecycleBaseCheap{};
        bool m_cachedNodesReady = false;

        std::vector<cudaGraphNode_t> m_velNodesFull,  m_velNodesCheap;
        std::vector<cudaKernelNodeParams> m_velNodeParamsBaseFull, m_velNodeParamsBaseCheap;
        bool m_cachedVelNodes = false;
        bool   m_velPatchedToDeltaFull = false;
        bool   m_velPatchedToDeltaCheap = false;

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

        // External semaphore for timeline fence
        cudaExternalSemaphore_t m_extTimelineSem = nullptr;
        uint64_t m_simFenceValue =0; // monotonically increasing simulation completion value

        // 渲染半精外部缓冲
        cudaExternalMemory_t m_extRenderHalf = nullptr;
        void* m_renderHalfMappedPtr = nullptr; // 指向 D3D12 half (uint2)资源的设备指针
        size_t m_renderHalfBytes =0;
    };
} // namespace sim