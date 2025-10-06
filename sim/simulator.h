#pragma once 
#include <cuda_runtime.h> 
#include "parameters.h" 
#include "device_buffers.cuh"
#include "stats.h" // 新增

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
        bool captureGraphIfNeeded(const SimParams& p);

        bool updateGridIfNeeded(const SimParams& p);
        bool needsGraphRebuild(const SimParams& p) const;

        void kIntegratePred(cudaStream_t s, const SimParams& p);
        void kHashKeys(cudaStream_t s, const SimParams& p);
        void kSort(cudaStream_t s, const SimParams& p);
        void kCellRanges(cudaStream_t s, const SimParams& p);
        void kSolveIter(cudaStream_t s, const SimParams& p);
        void kVelocityAndPost(cudaStream_t s, const SimParams& p);

    private:
        SimParams m_params{};
        DeviceBuffers m_bufs{};
        uint32_t m_numCells = 0;

        cudaStream_t m_stream = nullptr;
        cudaEvent_t  m_evStart = nullptr, m_evEnd = nullptr;

        cudaGraph_t m_graph = nullptr;
        cudaGraphExec_t m_graphExec = nullptr;
        bool m_graphDirty = true;

        int m_frameIndex = 0;

        // 记录上次捕获 Graph 的关键参数快照（用于判定是否需要重建）
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

        // 新增：CUDA 外部内存（映射 d_pos_pred）
        cudaExternalMemory_t m_extPosPred = nullptr;

        // 新增：回收内核对应的 Graph 节点（捕获后定位，逐帧改参数）
        cudaGraphNode_t m_nodeRecycle = nullptr;
    };
} // namespace sim