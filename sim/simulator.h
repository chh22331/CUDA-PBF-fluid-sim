#pragma once 
#include <cuda_runtime.h> 
#include "parameters.h" 
#include "device_buffers.cuh"
#include "stats.h" // 新增
#include <vector>

namespace sim {
    class Simulator {
    public:
        bool initialize(const SimParams& p);
        void shutdown();
        bool step(const SimParams& p);

        // 用预测/当前位置作为观测（去除 pos_pred->pos 拷贝）
        const float4* devicePositions() const { return m_bufs.d_pos_pred; }

        // 固定维度格点布点（xyz 方向点数）
        void seedBoxLattice(uint32_t nx, uint32_t ny, uint32_t nz, float3 origin, float spacing);

        // 新增：通用格点布点（任意 N 自动分解为 nx*ny*nz，受网格物理尺寸与 spacing 上限约束）
        void seedBoxLatticeAuto(uint32_t total, float3 origin, float spacing);

        // 新增：导入 D3D12 共享缓冲作为 d_pos_pred（sharedHandle 为 Win32 共享句柄）
        bool importPosPredFromD3D12(void* sharedHandleWin32, size_t bytes);

        // 新增：获取当前活动粒子数（随增长而变化）
        uint32_t activeParticleCount() const { return m_params.numParticles; }

        // 新增：统计接口（sampleStride=采样步长，>=1）
        bool computeStats(SimStats& out, uint32_t sampleStride = 4) const;

        // 新增：暴力统计（与 simulator.cpp 中的 LaunchComputeStatsBruteforce 对应）
        bool computeStatsBruteforce(SimStats& out, uint32_t sampleStride, uint32_t maxISamples) const;

        // 新增：采样质心（每 stride 取 1 个粒子，低成本）
        bool computeCenterOfMass(float3& outCom, uint32_t sampleStride) const;
    private:
        bool buildGrid(const SimParams& p);
        bool ensureSortTemp(std::size_t bytes);
        bool captureGraphIfNeeded(const SimParams& p); // 保留名，但内部构建两套 Graph

        bool updateGridIfNeeded(const SimParams& p);
        bool needsGraphRebuild(const SimParams& p) const;

        void kIntegratePred(cudaStream_t s, const SimParams& p);
        void kHashKeys(cudaStream_t s, const SimParams& p);
        void kSort(cudaStream_t s, const SimParams& p);
        void kCellRanges(cudaStream_t s, const SimParams& p);
        void kCellRangesCompact(cudaStream_t s, const SimParams& p);
        void kSolveIter(cudaStream_t s, const SimParams& p);
        void kVelocityAndPost(cudaStream_t s, const SimParams& p);

        // 新增：捕获后缓存 Graph 节点，避免每帧扫描
        bool cacheGraphNodes();

    private:
        SimParams m_params{};
        DeviceBuffers m_bufs{};
        uint32_t m_numCells = 0;

        bool     m_useHashedGrid = false;
        uint32_t m_numCompactCells = 0;

        cudaStream_t m_stream = nullptr;

        // —— 事件双缓冲非阻塞计时 —— //
        cudaEvent_t  m_evStart[2] = { nullptr, nullptr };
        cudaEvent_t  m_evEnd[2]   = { nullptr, nullptr };
        int          m_evCursor   = 0;
        float        m_lastFrameMs = -1.0f; // 最近一次成功读取到的 GPU 帧耗时

        // —— 双 Graph：Full（包含 Hash/Sort/Build），Cheap（省略这些阶段） —— //
        cudaGraph_t     m_graphFull = nullptr;
        cudaGraphExec_t m_graphExecFull = nullptr;
        cudaGraph_t     m_graphCheap = nullptr;
        cudaGraphExec_t m_graphExecCheap = nullptr;
        bool m_graphDirty = true;

        int m_frameIndex = 0;
        int m_lastFullFrame = -1; // 上一次运行 Full 的帧号（用于每 N 帧重建）

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

        // —— 缓存需要动态更新 gridDim 的节点（当前仅回收内核） —— //
        cudaGraphNode_t       m_nodeRecycleFull  = nullptr;
        cudaGraphNode_t       m_nodeRecycleCheap = nullptr;
        cudaKernelNodeParams  m_kpRecycleBaseFull{};
        cudaKernelNodeParams  m_kpRecycleBaseCheap{};
        bool                  m_cachedNodesReady = false;
    };
} // namespace sim