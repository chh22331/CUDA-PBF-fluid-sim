#pragma once
#include <cuda_runtime.h>
#include "parameters.h"
#include "device_buffers.cuh"
#include "grid_buffers.cuh"
#include "logging.h"
#include <vector>
#include "param_change_tracker.h"
#include "phase_pipeline.h"
#include "kernel_dispatcher.h"
#include "simulation_context.h"
#include "grid_strategy.h"
#include "post_ops.h"

namespace sim {

    // Simple online statistics accumulator (Welford's algorithm).
    // Tracks mean, running M2 and sample count for variance computation.
    struct OnlineVarHost {
        double mean = 0.0;
        double m2 = 0.0;
        uint64_t n = 0;

        // Add a new sample.
        void add(double x) {
            ++n;
            double d = x - mean;
            mean += d / double(n);
            double d2 = x - mean;
            m2 += d * d2;
        }

        // Return sample variance (0 if not enough samples).
        double variance() const {
            return (n > 1) ? (m2 / double(n - 1)) : 0.0;
        }
    };

    class GraphBuilder;

    // Simulator: manages particle data, CUDA resources, and execution graphs.
    // Responsibilities:
    //  - allocate and own Device/Grid buffers
    //  - build and manage CUDA graphs for position/velocity phases
    //  - seed/import particle data and expose buffers for rendering
    //  - track parameter changes and rebuild graphs when necessary
    class Simulator {
    public:
        // Initialize simulator resources based on parameters.
        bool initialize(const SimParams& p);

        // Release all GPU and host resources.
        void shutdown();

        // Execute one simulation step using the provided parameters.
        bool step(const SimParams& p);

        // Accessors for rendering / external use.
        const float4* devicePositions() const { return m_bufs.d_pos_curr; }
        float4* renderPositionPtr() const { return m_bufs.d_pos_curr; }

        // Population / import helpers.
        void seedBoxLattice(uint32_t nx, uint32_t ny, uint32_t nz, float3 origin, float spacing);
        void seedBoxLatticeAuto(uint32_t total, float3 origin, float spacing);
        bool importPosPredFromD3D12(void* sharedHandleWin32, size_t bytes);
        void performPingPongSwap(uint32_t N);
        bool bindExternalPosPingPong(void* sharedHandleA, size_t bytesA, void* sharedHandleB, size_t bytesB);
        uint32_t activeParticleCount() const { return m_params.numParticles; }
        bool computeCenterOfMass(float3& outCom, uint32_t sampleStride) const;
        double lastGpuFrameMs() const { return static_cast<double>(m_lastFrameMs); }
        void seedCubeMix(uint32_t groupCount, const float3* groupCenters, uint32_t edgeParticles, float spacing, bool applyJitter, float jitterAmp, uint32_t jitterSeed);
        float4* pingpongPosA() const { return (m_bufs.externalPingPong ? m_bufs.d_pos_curr : nullptr); }
        float4* pingpongPosB() const { return (m_bufs.externalPingPong ? m_bufs.d_pos_next : nullptr); }
        bool externalPingPongEnabled() const { return m_bufs.externalPingPong; }
        bool bindExternalVelocityBuffer(void* sharedHandle, size_t bytes, uint32_t strideBytes);
        void debugSampleDisplacement(uint32_t sampleStride = 1024);
        cudaStream_t cudaStream() const { return m_stream; }
        void syncForRender();
        bool swappedThisFrame() const { return m_swappedThisFrame; }

        // Timeline/fence for external sync (e.g. D3D12 interop).
        bool bindTimelineFence(HANDLE sharedFenceHandle);
        uint64_t lastSimFenceValue() const { return m_simFenceValue; }

        // Support for half-precision render buffer export/import.
        bool importRenderHalfBuffer(void* sharedHandleWin32, size_t bytes);
        void publishRenderHalf(uint32_t count);
        void releaseRenderHalfExternal();

        const float4* deviceVel() const { return m_bufs.d_vel; }

    private:
        friend class GraphBuilder;

        // Internal helpers for grid and graph management.
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
        void publishExternalVelocity(uint32_t count);

    private:
        // Current simulation parameters (source of truth for this instance).
        SimParams m_params{};

        // Device-side buffers for particle state.
        DeviceBuffers m_bufs{};

        // Grid bookkeeping buffers and counts.
        GridBuffers m_grid{};
        uint32_t m_numCells = 0;
        bool     m_useHashedGrid = false;
        uint32_t m_numCompactCells = 0;

        // CUDA runtime objects.
        cudaStream_t m_stream = nullptr;
        cudaEvent_t  m_evStart[2] = { nullptr, nullptr };
        cudaEvent_t  m_evEnd[2]   = { nullptr, nullptr };
        int          m_evCursor   = 0;

        // Frame timing / profiling.
        float        m_lastFrameMs = -1.0f;
        int          m_frameTimingEveryN = 1;
        bool         m_frameTimingEnabled = true;

        // Per-frame flags.
        bool         m_swappedThisFrame = false;

        // CUDA Graphs for efficient replay of kernel sequences.
        cudaGraph_t     m_graph = nullptr;
        cudaGraphExec_t m_graphExec = nullptr;
        bool m_graphDirty = true;
        bool m_paramDirty = true;
        bool m_canPingPongPos = true;
        bool m_graphPointersChecked = false;

        // Cached graph nodes and kernel parameter blocks for pos/vel phases.
        std::vector<cudaGraphNode_t>      m_posNodes;
        std::vector<cudaKernelNodeParams> m_posNodeParamsBase;
        bool m_cachedPosNodes = false;

        std::vector<cudaGraphNode_t>      m_velNodes;
        std::vector<cudaKernelNodeParams> m_velNodeParamsBase;
        bool m_cachedVelNodes = false;

        int  m_frameIndex = 0;
        int  m_lastParamUpdateFrame = -1;

        // Parameters captured at graph creation time used to detect when to rebuild.
        struct GraphCapturedParams {
            uint32_t numParticles = 0;
            uint32_t numCells = 0;
            int solverIters = 0;
            int maxNeighbors = 0;
            int sortEveryN = 1;
            GridBounds  grid{};
            KernelCoeffs kernel{};
            float dt = 0.0f;
            float3 gravity = make_float3(0.f, 0.f, 0.f);
            float restDensity = 0.0f;
        } m_captured{};

        // External memory handles for interop (e.g. shared D3D12 buffers).
        cudaExternalMemory_t m_extPosPred = nullptr;
        cudaExternalMemory_t m_extraExternalMemB = nullptr;
        cudaExternalMemory_t m_extVelocityMem = nullptr;
        void*  m_extVelocityPtr = nullptr;
        size_t m_extVelocityBytes = 0;
        uint32_t m_extVelocityStride = 0;

        // Node recycling to avoid repeated allocations when rebuilding graphs.
        cudaGraphNode_t      m_nodeRecycle  = nullptr;
        cudaKernelNodeParams m_kpRecycleBase{};
        bool m_cachedNodesReady = false;

        // Adaptive scheme history for solver tuning.
        OnlineVarHost m_adaptDensityErrorHistory{};
        OnlineVarHost m_adaptLambdaVarHistory{};
        int  m_adaptUpgradeHold = 0;
        int  m_adaptDowngradeHold = 0;
        bool m_adaptHalfDisabled = false;

        // Helpers and pipelines composing the simulation execution.
        ParamChangeTracker m_paramTracker;
        KernelDispatcher   m_kernelDispatcher;
        PhasePipeline      m_pipeline;
        SimulationContext  m_ctx;
        std::unique_ptr<IGridStrategy> m_gridStrategy;
        PostOpsPipeline    m_postPipeline;

        // External timeline semaphore for cross-API synchronization.
        cudaExternalSemaphore_t m_extTimelineSem = nullptr;
        uint64_t m_simFenceValue = 0;

        // External half-precision render buffer bookkeeping.
        cudaExternalMemory_t m_extRenderHalf = nullptr;
        void*  m_renderHalfMappedPtr = nullptr;
        size_t m_renderHalfBytes = 0;
    };
} // namespace sim