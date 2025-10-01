#include "simulator.h"
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

// —— 内核包装的全局声明（与 .cu 文件保持一致的 extern "C" + 全局命名空间） ——
extern "C" void LaunchIntegratePred(float4* pos, const float4* vel, float4* pos_pred, float3 gravity, float dt, uint32_t N, cudaStream_t s);
extern "C" void LaunchHashKeys(uint32_t* keys, uint32_t* indices, const float4* pos, sim::GridBounds grid, uint32_t N, cudaStream_t s);
extern "C" void LaunchCellRanges(uint32_t* cellStart, uint32_t* cellEnd, const uint32_t* keysSorted, uint32_t N, uint32_t numCells, cudaStream_t s);
extern "C" void LaunchLambda(float* lambda, const float4* pos_pred, const uint32_t* indicesSorted, const uint32_t* cellStart, const uint32_t* cellEnd, sim::GridBounds grid, sim::KernelCoeffs kc, float restDensity, int maxNeighbors, uint32_t N, cudaStream_t s);
extern "C" void LaunchDeltaApply(float4* pos_pred, float4* delta, const float* lambda, const uint32_t* indicesSorted, const uint32_t* cellStart, const uint32_t* cellEnd, sim::GridBounds grid, sim::KernelCoeffs kc, float restDensity, int maxNeighbors, uint32_t N, cudaStream_t s);
extern "C" void LaunchVelocity(float4* vel, const float4* pos, const float4* pos_pred, float inv_dt, uint32_t N, cudaStream_t s);
extern "C" void LaunchBoundary(float4* pos_pred, float4* vel, sim::GridBounds grid, uint32_t N, cudaStream_t s);

// —— CUB radix sort（与 sort_pairs.cu 完全一致的 C 接口） ——
extern "C" void LaunchSortPairsQuery(size_t* tempBytes,
                                     const uint32_t* d_keys_in, uint32_t* d_keys_out,
                                     const uint32_t* d_vals_in, uint32_t* d_vals_out,
                                     uint32_t N, cudaStream_t s);
extern "C" void LaunchSortPairs(void* d_temp_storage, size_t tempBytes,
                                uint32_t* d_keys_in, uint32_t* d_keys_out,
                                uint32_t* d_vals_in, uint32_t* d_vals_out,
                                uint32_t N, cudaStream_t s);

namespace sim {

    bool Simulator::initialize(const SimParams& p) {
        m_params = p;
        CUDA_CHECK(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&m_evStart));
        CUDA_CHECK(cudaEventCreate(&m_evEnd));

        // 分配主状态与邻域所需缓冲（容量）
        m_bufs.allocate(p.numParticles);

        // 初始化网格（CellStart/End 尺寸）
        if (!buildGrid(p)) return false;

        // 初始化 indices = [0..N)
        std::vector<uint32_t> h_idx(p.numParticles);
        for (uint32_t i = 0; i < p.numParticles; ++i) h_idx[i] = i;
        CUDA_CHECK(cudaMemcpy(m_bufs.d_indices, h_idx.data(), sizeof(uint32_t) * p.numParticles, cudaMemcpyHostToDevice));

        // 确保 pos_pred 初始与 pos 同步（首帧）
        CUDA_CHECK(cudaMemcpy(m_bufs.d_pos_pred, m_bufs.d_pos, sizeof(float4) * p.numParticles, cudaMemcpyDeviceToDevice));

        m_graphDirty = true;
        return true;
    }

    void Simulator::shutdown() {
        if (m_graphExec) { cudaGraphExecDestroy(m_graphExec); m_graphExec = nullptr; }
        if (m_graph)     { cudaGraphDestroy(m_graph); m_graph = nullptr; }
        if (m_evStart)   { cudaEventDestroy(m_evStart); m_evStart = nullptr; }
        if (m_evEnd)     { cudaEventDestroy(m_evEnd);   m_evEnd = nullptr; }
        if (m_stream)    { cudaStreamDestroy(m_stream); m_stream = nullptr; }
    }

    bool Simulator::buildGrid(const SimParams& p) {
        int3 dim;
        dim.x = int(ceilf((p.grid.maxs.x - p.grid.mins.x) / p.grid.cellSize));
        dim.y = int(ceilf((p.grid.maxs.y - p.grid.mins.y) / p.grid.cellSize));
        dim.z = int(ceilf((p.grid.maxs.z - p.grid.mins.z) / p.grid.cellSize));
        m_numCells = uint32_t(dim.x) * uint32_t(dim.y) * uint32_t(dim.z);
        if (m_numCells == 0) return false;
        m_bufs.allocGridRanges(m_numCells);
        return true;
    }

    bool Simulator::ensureSortTemp(std::size_t bytes) {
        m_bufs.ensureSortTemp(bytes);
        return true;
    }

    // 捕获整步 CUDA Graph（包含 Integrate→Hash/Sort→CellRanges→K 次迭代→Velocity/Boundary）
    bool Simulator::captureGraphIfNeeded(const SimParams& p) {
        if (!m_graphDirty) return true;
        if (m_graphExec) { cudaGraphExecDestroy(m_graphExec); m_graphExec = nullptr; }
        if (m_graph)     { cudaGraphDestroy(m_graph); m_graph = nullptr; }

        CUDA_CHECK(cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeGlobal));

        kIntegratePred(m_stream, p);
        kHashKeys(m_stream, p);
        kSort(m_stream, p);
        kCellRanges(m_stream, p);

        for (int i = 0; i < p.solverIters; ++i) {
            kSolveIter(m_stream, p);
        }
        kVelocityAndPost(m_stream, p);

        CUDA_CHECK(cudaStreamEndCapture(m_stream, &m_graph));
        CUDA_CHECK(cudaGraphInstantiate(&m_graphExec, m_graph, nullptr, nullptr, 0));
        m_graphDirty = false;
        return true;
    }

    bool Simulator::step(const SimParams& p) {
        m_params = p;

        // 容量扩展（粒子数增长时）
        if (m_params.numParticles > m_bufs.capacity) {
            m_bufs.allocate(m_params.numParticles);
            // 重新分配后 indices 需要重置
            std::vector<uint32_t> h_idx(m_params.numParticles);
            for (uint32_t i = 0; i < m_params.numParticles; ++i) h_idx[i] = i;
            CUDA_CHECK(cudaMemcpy(m_bufs.d_indices, h_idx.data(), sizeof(uint32_t) * m_params.numParticles, cudaMemcpyHostToDevice));
            m_graphDirty = true;
        }

        if (!captureGraphIfNeeded(m_params)) return false;

        CUDA_CHECK(cudaEventRecord(m_evStart, m_stream));
        CUDA_CHECK(cudaGraphLaunch(m_graphExec, m_stream));
        CUDA_CHECK(cudaEventRecord(m_evEnd, m_stream));
        CUDA_CHECK(cudaEventSynchronize(m_evEnd));

        float ms = 0.f; cudaEventElapsedTime(&ms, m_evStart, m_evEnd);
        ++m_frameIndex;
        return true;
    }

    // —— 阶段封装 ——
    void Simulator::kIntegratePred(cudaStream_t s, const SimParams& p) {
        // 积分在 pos_pred 上就地进行（避免 pos_pred->pos 的整缓冲拷贝）
        ::LaunchIntegratePred(m_bufs.d_pos_pred, m_bufs.d_vel, m_bufs.d_pos_pred, p.gravity, p.dt, p.numParticles, s);
    }

    void Simulator::kHashKeys(cudaStream_t s, const SimParams& p) {
        ::LaunchHashKeys(m_bufs.d_cellKeys, m_bufs.d_indices, m_bufs.d_pos_pred, p.grid, p.numParticles, s);
    }

    void Simulator::kSort(cudaStream_t s, const SimParams& p) {
        size_t tempBytes = 0;
        ::LaunchSortPairsQuery(&tempBytes,
            m_bufs.d_cellKeys, m_bufs.d_cellKeys_sorted,
            m_bufs.d_indices,  m_bufs.d_indices_sorted,
            p.numParticles, s);

        ensureSortTemp(tempBytes);

        ::LaunchSortPairs(m_bufs.d_sortTemp, tempBytes,
            m_bufs.d_cellKeys, m_bufs.d_cellKeys_sorted,
            m_bufs.d_indices,  m_bufs.d_indices_sorted,
            p.numParticles, s);
    }

    void Simulator::kCellRanges(cudaStream_t s, const SimParams& p) {
        ::LaunchCellRanges(m_bufs.d_cellStart, m_bufs.d_cellEnd, m_bufs.d_cellKeys_sorted, p.numParticles, m_numCells, s);
    }

    void Simulator::kSolveIter(cudaStream_t s, const SimParams& p) {
        ::LaunchLambda(m_bufs.d_lambda, m_bufs.d_pos_pred, m_bufs.d_indices_sorted,
            m_bufs.d_cellStart, m_bufs.d_cellEnd, p.grid, p.kernel, p.restDensity, p.maxNeighbors, p.numParticles, s);

        ::LaunchDeltaApply(m_bufs.d_pos_pred, m_bufs.d_delta, m_bufs.d_lambda, m_bufs.d_indices_sorted,
            m_bufs.d_cellStart, m_bufs.d_cellEnd, p.grid, p.kernel, p.restDensity, p.maxNeighbors, p.numParticles, s);
    }

    void Simulator::kVelocityAndPost(cudaStream_t s, const SimParams& p) {
        ::LaunchBoundary(m_bufs.d_pos_pred, m_bufs.d_vel, p.grid, p.numParticles, s);
        ::LaunchVelocity(m_bufs.d_vel, m_bufs.d_pos, m_bufs.d_pos_pred, 1.0f / p.dt, p.numParticles, s);
        // 去除不必要的整缓冲 D2D 拷贝；渲染侧读取 devicePositions()（即 d_pos_pred）
        // CUDA_CHECK(cudaMemcpyAsync(m_bufs.d_pos, m_bufs.d_pos_pred, sizeof(float4) * p.numParticles, cudaMemcpyDeviceToDevice, s));
    }

} // namespace sim