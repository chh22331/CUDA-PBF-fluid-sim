#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>              // 新增: std::size_t
#include "parameters.h"
#include "device_buffers.cuh"

namespace sim {

    class Simulator {
    public:
        bool initialize(const SimParams& p);
        void shutdown();

        // 每帧：可选更新参数（dt/cfl/K 等），然后 step
        bool step(const SimParams& p);

        // 导出给 D3D12 的粒子位置（若使用共享 External Memory，可直接返回 device ptr）
        const float4* devicePositions() const { return m_bufs.d_pos; }

    private:
        bool buildGrid(const SimParams& p);
        bool captureGraphIfNeeded(const SimParams& p);
        bool ensureSortTemp(std::size_t bytes);

        // 阶段函数
        void kIntegratePred(cudaStream_t s, const SimParams& p);
        void kHashKeys(cudaStream_t s, const SimParams& p);
        void kSort(cudaStream_t s, const SimParams& p);
        void kCellRanges(cudaStream_t s, const SimParams& p);
        void kSolveIter(cudaStream_t s, const SimParams& p);
        void kVelocityAndPost(cudaStream_t s, const SimParams& p);

    private:
        SimParams m_params{};
        DeviceBuffers m_bufs{};
        std::uint32_t m_numCells = 0;

        cudaStream_t m_stream = nullptr;
        cudaEvent_t  m_evStart = nullptr, m_evEnd = nullptr;

        // Graph
        cudaGraph_t m_graph = nullptr;
        cudaGraphExec_t m_graphExec = nullptr;
        bool m_graphDirty = true;

        // 统计
        int m_frameIndex = 0;
    };

} // namespace sim
