#pragma once
#include <cuda_runtime.h>
#include "parameters.h"
#include "device_buffers.cuh"
#include "grid_buffers.cuh"
#include "emit_params.h"
#include "emitter.h"
#include "logging.h"
#include "stats.h" // 修复：需要 SimStats 定义
#include <vector>
#include "param_change_tracker.h" // new tracker
#include "phase_pipeline.h"
#include "kernel_dispatcher.h"
#include "simulation_context.h"
#include "grid_strategy.h"
#include "post_ops.h"

namespace sim {
    struct OnlineVarHost {
        double mean = 0.0;
        double m2 = 0.0;
        uint64_t n = 0;
        void add(double x) {
            ++n;
            double d = x - mean;
            mean += d / double(n);
            double d2 = x - mean;
            m2 += d * d2;
        }
        double variance() const { return (n > 1) ? (m2 / double(n - 1)) : 0.0; }
    };

    class Simulator {
    public:
        bool initialize(const SimParams& p);
        void shutdown();
        bool step(const SimParams& p);

        const float4* devicePositions() const { return m_bufs.d_pos_pred; }

        void seedBoxLattice(uint32_t nx, uint32_t ny, uint32_t nz, float3 origin, float spacing);
        void seedBoxLatticeAuto(uint32_t total, float3 origin, float spacing);
        bool importPosPredFromD3D12(void* sharedHandleWin32, size_t bytes);
        bool bindExternalPosPingPong(void* sharedHandleA, size_t bytesA, void* sharedHandleB, size_t bytesB); // new API
        uint32_t activeParticleCount() const { return m_params.numParticles; }
        bool computeStats(SimStats& out, uint32_t sampleStride = 4) const;
        bool computeStatsBruteforce(SimStats& out, uint32_t sampleStride, uint32_t maxISamples) const;
        bool computeCenterOfMass(float3& outCom, uint32_t sampleStride) const;
        double lastGpuFrameMs() const { return static_cast<double>(m_lastFrameMs); }
        void seedCubeMix(uint32_t groupCount,
            const float3* groupCenters,
            uint32_t edgeParticles,
            float spacing,
            bool applyJitter,
            float jitterAmp,
            uint32_t jitterSeed);
        bool adaptivePrecisionCheck(const SimStats& stats);
        bool tryPingPongAndUpdatePtrs(bool useGraphs);

        float4* pingpongPosA() const { return (m_bufs.externalPingPong ? m_bufs.d_pos_curr : nullptr); } // 在 bind 后初始 curr=A
        float4* pingpongPosB() const { return (m_bufs.externalPingPong ? m_bufs.d_pos_next : nullptr); }
        bool externalPingPongEnabled() const { return m_bufs.externalPingPong; }
        void debugSampleDisplacement(uint32_t sampleStride = 1024);

        cudaStream_t cudaStream() const { return m_stream; }
        void syncForRender(); // 新增

    private:
        bool buildGrid(const SimParams& p);
        bool ensureSortTemp(std::size_t bytes);
        bool structuralGraphChange(const SimParams& p) const; // legacy to be replaced by tracker
        bool paramOnlyGraphChange(const SimParams& p) const;  // legacy to be replaced
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
        bool patchGraphKernelPointers(bool fullGraph);
        bool attemptGraphPingPongAndPatch(bool useGraphs, bool needFull);


    private:
        SimParams m_params{};
        DeviceBuffers m_bufs{};        // 粒子主/半精缓冲
        GridBuffers  m_grid{};         // 网格/排序/压缩缓冲（拆分）
        uint32_t m_numCells = 0;
        bool     m_useHashedGrid = false;
        uint32_t m_numCompactCells = 0;
        cudaStream_t m_stream = nullptr;
        cudaEvent_t  m_evStart[2] = { nullptr, nullptr };
        cudaEvent_t  m_evEnd[2] = { nullptr, nullptr };
        int          m_evCursor = 0;
        float        m_lastFrameMs = -1.0f;
        int   m_frameTimingEveryN = 1;
        bool  m_frameTimingEnabled = true;
        friend class GraphBuilder;
        cudaEvent_t m_evGraphEnd = nullptr;

        cudaGraph_t     m_graphFull = nullptr;
        cudaGraphExec_t m_graphExecFull = nullptr;
        cudaGraph_t     m_graphCheap = nullptr;
        cudaGraphExec_t m_graphExecCheap = nullptr;
        bool m_graphDirty = true;
        bool m_paramDirty = true;
        bool m_canPingPongPos = true;
        bool m_precisionLogged = false;
        bool m_graphPointersChecked = false;
        bool m_lastFrameXsphApplied = false;   // 新增：记录上一帧是否使用 XSPH
        bool m_graphNodesPatchedOnce = false;  // 新增：至少执行过一次 patch


        std::vector<cudaGraphNode_t> m_posNodesFull, m_posNodesCheap;
        std::vector<cudaKernelNodeParams> m_posNodeParamsBaseFull, m_posNodeParamsBaseCheap;
        bool m_cachedPosNodes = false;

        int  m_frameIndex = 0;
        int  m_lastFullFrame = -1;
        int  m_lastParamUpdateFrame = -1;
        struct GraphCapturedParams {
            uint32_t numParticles = 0;
            uint32_t numCells = 0;
            int      solverIters = 0;
            int      maxNeighbors = 0;
            int      sortEveryN = 1;
            GridBounds  grid{};
            KernelCoeffs kernel{};
            float    dt = 0.0f;
            float3   gravity = make_float3(0.f, 0.f, 0.f);
            float    restDensity = 0.0f;
        } m_captured{};
        cudaExternalMemory_t m_extPosPred = nullptr;

        cudaGraphNode_t       m_nodeRecycleFull = nullptr;
        cudaGraphNode_t       m_nodeRecycleCheap = nullptr;
        cudaKernelNodeParams  m_kpRecycleBaseFull{};
        cudaKernelNodeParams  m_kpRecycleBaseCheap{};
        bool                  m_cachedNodesReady = false;

        std::vector<cudaGraphNode_t> m_velNodesFull, m_velNodesCheap;
        std::vector<cudaKernelNodeParams> m_velNodeParamsBaseFull, m_velNodeParamsBaseCheap;
        bool m_cachedVelNodes = false;

        // 自适应精度状态
        bool m_adaptHalfDisabled = false;
        int  m_adaptUpgradeHold = 0;
        int  m_adaptDowngradeHold = 0;
        OnlineVarHost m_adaptDensityErrorHistory;
        OnlineVarHost m_adaptLambdaVarHistory;

        // === New components under refactor ===
        ParamChangeTracker m_paramTracker; // unified change detection
        KernelDispatcher   m_kernelDispatcher; // kernel selection
        PhasePipeline      m_pipeline; // full/cheap sequences
        SimulationContext  m_ctx; // shared runtime view
        std::unique_ptr<IGridStrategy> m_gridStrategy; // hashed or dense
        PostOpsPipeline m_postPipeline; // new post ops
    };
} // namespace sim