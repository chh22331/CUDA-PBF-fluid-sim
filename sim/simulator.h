#pragma once
#include <cuda_runtime.h>
#include "parameters.h"
#include "device_buffers.cuh"

namespace sim {

class Simulator {
public:
    bool initialize(const SimParams& p);
    void shutdown();
    bool step(const SimParams& p);

    // 用预测/当前位置作为观测（去除 pos_pred->pos 拷贝）
    const float4* devicePositions() const { return m_bufs.d_pos_pred; }

    void seedBoxLattice(uint32_t nx, uint32_t ny, uint32_t nz, float3 origin, float spacing);

private:
    bool buildGrid(const SimParams& p);
    bool ensureSortTemp(std::size_t bytes);
    bool captureGraphIfNeeded(const SimParams& p);

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
};

} // namespace sim
