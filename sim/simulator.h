#pragma once
#include <cuda_runtime.h>
#include "parameters.h"
#include "device_buffers.cuh"
#include "stats.h"
#include <vector>

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
        inline const DeviceBuffers& buffersUnsafe() const { return m_bufs; }
        inline DeviceBuffers& buffersUnsafe() { return m_bufs; }
        inline uint64_t frameIndexUnsafe() const { return m_frameIndex; }
        bool updateGraphPointersAfterSwap(const void* oldPos,
            const void* oldPosPred,
            const void* oldVel,
            const void* oldDelta,
            bool swappedVel);

    private:
        bool buildGrid(const SimParams& p);
        bool ensureSortTemp(std::size_t bytes);
        bool structuralGraphChange(const SimParams& p) const;
        bool paramOnlyGraphChange(const SimParams& p) const;
        bool captureGraphIfNeeded(const SimParams& p);
        bool updateGraphsParams(const SimParams& p);
        bool updateGridIfNeeded(const SimParams& p);
        void kIntegratePred(cudaStream_t s, const SimParams& p);
        void kSnapshotPrevPos(cudaStream_t s, const SimParams& p);
        void kHashKeys(cudaStream_t s, const SimParams& p);
        void kSort(cudaStream_t s, const SimParams& p);
        void kCellRanges(cudaStream_t s, const SimParams& p);
        void kCellRangesCompact(cudaStream_t s, const SimParams& p);
        void kSolveIter(cudaStream_t s, const SimParams& p);
        void kVelocityAndPost(cudaStream_t s, const SimParams& p);
        void kRestorePosFromSnapshot(cudaStream_t s, const SimParams& p); // 新增
        bool cacheGraphNodes();

    private:
        SimParams m_params{};
        DeviceBuffers m_bufs{};
        uint32_t m_numCells = 0;
        bool     m_useHashedGrid = false;
        uint32_t m_numCompactCells = 0;
        cudaStream_t m_stream = nullptr;
        cudaEvent_t  m_evStart[2] = { nullptr, nullptr };
        cudaEvent_t  m_evEnd[2]   = { nullptr, nullptr };
        int          m_evCursor   = 0;
        float        m_lastFrameMs = -1.0f;
        int   m_frameTimingEveryN = 1;
        bool  m_frameTimingEnabled = true;

        cudaGraph_t     m_graphFull = nullptr;
        cudaGraphExec_t m_graphExecFull = nullptr;
        cudaGraph_t     m_graphCheap = nullptr;
        cudaGraphExec_t m_graphExecCheap = nullptr;
        bool m_graphDirty = true;
        bool m_paramDirty = true;
        bool m_canPingPongPos = true;
        bool m_precisionLogged = false;
        bool m_graphPointersChecked = false;

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

        cudaGraphNode_t       m_nodeRecycleFull  = nullptr;
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
    };
} // namespace sim