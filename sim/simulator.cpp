#include "simulator.h"
#include <cstdio>
#include <vector>     // 新增: std::vector
#include <cmath>      // 新增: ceilf
#include <cstdint>    // 可选: 定宽整数（头文件已包含，此处冗余但安全）

namespace sim {

    // ————— 内核声明（见 kernels/*.cu） —————
    void LaunchIntegratePred(float4* pos, const float4* vel, float4* pos_pred, float3 gravity, float dt, std::uint32_t N, cudaStream_t s);
    void LaunchHashKeys(std::uint32_t* keys, std::uint32_t* indices, const float4* pos, GridBounds grid, std::uint32_t N, cudaStream_t s);
    void LaunchCellRanges(std::uint32_t* cellStart, std::uint32_t* cellEnd, const std::uint32_t* keysSorted, std::uint32_t N, std::uint32_t numCells, cudaStream_t s);
    void LaunchLambda(float* lambda, const float4* pos_pred, const std::uint32_t* indicesSorted, const std::uint32_t* cellStart, const std::uint32_t* cellEnd, GridBounds grid, KernelCoeffs kc, float restDensity, int maxNeighbors, std::uint32_t N, cudaStream_t s);
    void LaunchDeltaApply(float4* pos_pred, float4* delta, const float* lambda, const std::uint32_t* indicesSorted, const std::uint32_t* cellStart, const std::uint32_t* cellEnd, GridBounds grid, KernelCoeffs kc, int maxNeighbors, std::uint32_t N, cudaStream_t s);
    void LaunchVelocity(float4* vel, const float4* pos, const float4* pos_pred, float inv_dt, std::uint32_t N, cudaStream_t s);
    void LaunchBoundary(float4* pos_pred, float4* vel, GridBounds grid, std::uint32_t N, cudaStream_t s);

    // ————— CUB radix sort 封装（在 .cu 中实现，由 nvcc 编译） —————
    extern "C" void LaunchSortPairsQuery(std::size_t* tempBytes,
                                         const std::uint32_t* d_keys_in, std::uint32_t* d_keys_out,
                                         const std::uint32_t* d_vals_in, std::uint32_t* d_vals_out,
                                         std::uint32_t N, cudaStream_t s);
    extern "C" void LaunchSortPairs(void* d_temp_storage, std::size_t tempBytes,
                                    std::uint32_t* d_keys_in, std::uint32_t* d_keys_out,
                                    std::uint32_t* d_vals_in, std::uint32_t* d_vals_out,
                                    std::uint32_t N, cudaStream_t s);

    // ————— Simulator 实现 —————
    bool Simulator::initialize(const SimParams& p) {
        m_params = p;
        CUDA_CHECK(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&m_evStart));
        CUDA_CHECK(cudaEventCreate(&m_evEnd));

        m_bufs.allocate(p.numParticles);

        // grid 尺寸与 cell 数
        if (!buildGrid(p)) return false;

        // indices 初始化 [0..N)
        std::vector<std::uint32_t> h_idx(p.numParticles);
        for (std::uint32_t i = 0; i < p.numParticles; ++i) h_idx[i] = i;
        CUDA_CHECK(cudaMemcpy(m_bufs.d_indices, h_idx.data(), sizeof(std::uint32_t) * p.numParticles, cudaMemcpyHostToDevice));

        m_graphDirty = true;
        return true;
    }

    void Simulator::shutdown() {
        if (m_graphExec) { cudaGraphExecDestroy(m_graphExec); m_graphExec = nullptr; }
        if (m_graph) { cudaGraphDestroy(m_graph); m_graph = nullptr; }
        if (m_evStart) cudaEventDestroy(m_evStart);
        if (m_evEnd)   cudaEventDestroy(m_evEnd);
        if (m_stream)  cudaStreamDestroy(m_stream);
    }

    bool Simulator::buildGrid(const SimParams& p) {
        int3 dim;
        dim.x = int(ceilf((p.grid.maxs.x - p.grid.mins.x) / p.grid.cellSize));
        dim.y = int(ceilf((p.grid.maxs.y - p.grid.mins.y) / p.grid.cellSize));
        dim.z = int(ceilf((p.grid.maxs.z - p.grid.mins.z) / p.grid.cellSize));
        m_numCells = std::uint32_t(dim.x) * std::uint32_t(dim.y) * std::uint32_t(dim.z);
        if (m_numCells == 0) return false;
        m_bufs.allocGridRanges(m_numCells);
        return true;
    }

    bool Simulator::ensureSortTemp(std::size_t bytes) {
        m_bufs.ensureSortTemp(bytes);
        return true;
    }

    bool Simulator::captureGraphIfNeeded(const SimParams& p) {
        if (!m_graphDirty) return true;
        if (m_graphExec) { cudaGraphExecDestroy(m_graphExec); m_graphExec = nullptr; }
        if (m_graph) { cudaGraphDestroy(m_graph); m_graph = nullptr; }

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
        if (m_params.numParticles > m_bufs.capacity) m_bufs.allocate(m_params.numParticles);
        if (!captureGraphIfNeeded(m_params)) return false;

        CUDA_CHECK(cudaEventRecord(m_evStart, m_stream));
        CUDA_CHECK(cudaGraphLaunch(m_graphExec, m_stream));
        CUDA_CHECK(cudaEventRecord(m_evEnd, m_stream));
        CUDA_CHECK(cudaEventSynchronize(m_evEnd));

        float ms = 0.f; cudaEventElapsedTime(&ms, m_evStart, m_evEnd);
        ++m_frameIndex;
        return true;
    }

    // ————— 阶段封装 —————
    void Simulator::kIntegratePred(cudaStream_t s, const SimParams& p) {
        LaunchIntegratePred(m_bufs.d_pos, m_bufs.d_vel, m_bufs.d_pos_pred, p.gravity, p.dt, p.numParticles, s);
    }

    void Simulator::kHashKeys(cudaStream_t s, const SimParams& p) {
        LaunchHashKeys(m_bufs.d_cellKeys, m_bufs.d_indices, m_bufs.d_pos_pred, p.grid, p.numParticles, s);
    }

    void Simulator::kSort(cudaStream_t s, const SimParams& p) {
        // 使用 CUB radix sort（由 .cu 封装）
        std::size_t tempBytes = 0;
        LaunchSortPairsQuery(&tempBytes,
            m_bufs.d_cellKeys, m_bufs.d_cellKeys,       // keys in/out（就地）
            m_bufs.d_indices, m_bufs.d_indices_sorted,  // vals in/out
            p.numParticles, s);

        ensureSortTemp(tempBytes);

        LaunchSortPairs(m_bufs.d_sortTemp, tempBytes,
            m_bufs.d_cellKeys, m_bufs.d_cellKeys,
            m_bufs.d_indices, m_bufs.d_indices_sorted,
            p.numParticles, s);
    }

    void Simulator::kCellRanges(cudaStream_t s, const SimParams& p) {
        LaunchCellRanges(m_bufs.d_cellStart, m_bufs.d_cellEnd, m_bufs.d_cellKeys, p.numParticles, m_numCells, s);
    }

    void Simulator::kSolveIter(cudaStream_t s, const SimParams& p) {
        LaunchLambda(m_bufs.d_lambda, m_bufs.d_pos_pred, m_bufs.d_indices_sorted,
            m_bufs.d_cellStart, m_bufs.d_cellEnd, p.grid, p.kernel, p.restDensity, p.maxNeighbors, p.numParticles, s);

        LaunchDeltaApply(m_bufs.d_pos_pred, m_bufs.d_delta, m_bufs.d_lambda, m_bufs.d_indices_sorted,
            m_bufs.d_cellStart, m_bufs.d_cellEnd, p.grid, p.kernel, p.maxNeighbors, p.numParticles, s);
    }

    void Simulator::kVelocityAndPost(cudaStream_t s, const SimParams& p) {
        LaunchBoundary(m_bufs.d_pos_pred, m_bufs.d_vel, p.grid, p.numParticles, s);
        LaunchVelocity(m_bufs.d_vel, m_bufs.d_pos, m_bufs.d_pos_pred, 1.0f / p.dt, p.numParticles, s);
        CUDA_CHECK(cudaMemcpyAsync(m_bufs.d_pos, m_bufs.d_pos_pred, sizeof(float4) * p.numParticles, cudaMemcpyDeviceToDevice, s));
    }

} // namespace sim