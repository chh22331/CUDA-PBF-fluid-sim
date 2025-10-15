#include "simulator.h" 
#include "emit_params.h" 
#include "stats.h"
#include <cstdio> 
#include <vector> 
#include <cmath> 
#include <cstdint> 
#include <cstddef> 
#include <random> 
#include <algorithm>
#include <cuda_runtime.h>
#include <unordered_set>
#include "../engine/core/console.h"
#include "../engine/core/prof_nvtx.h"
#include "precision_traits.cuh"
#include "precision_stage.h"
#include <limits>

#ifndef CUDA_CHECK
#define CUDA_CHECK(expr) \
    do { \
        cudaError_t _err = (expr); \
        if (_err != cudaSuccess) { \
            std::fprintf(stderr, "[CUDA] %s:%d %s (%d)\n", __FILE__, __LINE__, cudaGetErrorString(_err), (int)_err); \
        } \
    } while (0)
#endif

// —— 内核包装的全局声明 —— 
extern "C" void LaunchIntegratePred(float4* pos, const float4* vel, float4* pos_pred, float3 gravity, float dt, uint32_t N, cudaStream_t s);
extern "C" void LaunchIntegratePredMP(const float4 * pos, const float4 * vel, float4 * pos_pred,
    const sim::Half4 * pos_h4, const sim::Half4 * vel_h4,
    float3 gravity, float dt, uint32_t N, cudaStream_t s);

extern "C" void LaunchHashKeys(uint32_t* keys, uint32_t* indices, const float4* pos, sim::GridBounds grid, uint32_t N, cudaStream_t s);
extern "C" void LaunchHashKeysMP(uint32_t * keys, uint32_t * indices, const float4 * pos_pred,
    const sim::Half4 * pos_pred_h4, sim::GridBounds grid, uint32_t N, cudaStream_t s);
extern "C" void LaunchCellRanges(uint32_t* cellStart, uint32_t* cellEnd, const uint32_t* keysSorted, uint32_t N, uint32_t numCells, cudaStream_t s);
extern "C" void LaunchCellRangesCompact(
    uint32_t* d_uniqueKeys,
    uint32_t* d_offsets,
    uint32_t* d_compactCount,
    const uint32_t* d_keysSorted,
    uint32_t N,
    cudaStream_t s);

extern "C" void LaunchLambda(
    float* lambda,
    const float4* pos_pred,
    const uint32_t* indicesSorted,
    const uint32_t* keysSorted,
    const uint32_t* cellStart,
    const uint32_t* cellEnd,
    sim::DeviceParams dp,
    uint32_t N,
    cudaStream_t s);
extern "C" void LaunchLambdaCompact(
    float* lambda,
    const float4* pos_pred,
    const uint32_t* indicesSorted,
    const uint32_t* keysSorted,
    const uint32_t* uniqueKeys,
    const uint32_t* offsets,
    const uint32_t* compactCount,
    sim::DeviceParams dp,
    uint32_t N,
    cudaStream_t s);
extern "C" void LaunchLambdaCompactMP(
    float* lambda,
    const float4 * pos_pred_fp32, const sim::Half4 * pos_pred_h4,
    const uint32_t * indicesSorted, const uint32_t * keysSorted,
    const uint32_t * uniqueKeys, const uint32_t * offsets,
    const uint32_t * compactCount,
    sim::DeviceParams dp, uint32_t N, cudaStream_t s);

extern "C" void LaunchDeltaApply(
    float4* pos_pred,
    float4* delta,
    const float* lambda,
    const uint32_t* indicesSorted,
    const uint32_t* keysSorted,
    const uint32_t* cellStart,
    const uint32_t* cellEnd,
    sim::DeviceParams dp,
    uint32_t N,
    cudaStream_t s);
extern "C" void LaunchDeltaApplyCompact(
    float4* pos_pred,
    float4* delta,
    const float* lambda,
    const uint32_t* indicesSorted,
    const uint32_t* keysSorted,
    const uint32_t* uniqueKeys,
    const uint32_t* offsets,
    const uint32_t* compactCount,
    sim::DeviceParams dp,
    uint32_t N,
    cudaStream_t s);
extern "C" void LaunchDeltaApplyCompactMP(
    float4 * pos_pred,
    float4 * delta,
    const float* lambda,
    const float4 * pos_pred_fp32, const sim::Half4 * pos_pred_h4,
    const uint32_t * indicesSorted, const uint32_t * keysSorted,
    const uint32_t * uniqueKeys, const uint32_t * offsets,
    const uint32_t * compactCount,
    sim::DeviceParams dp, uint32_t N, cudaStream_t s);
extern "C" void LaunchXSPH(
    float4* vel_out,
    const float4* vel_in,
    const float4* pos_pred,
    const uint32_t* indicesSorted,
    const uint32_t* keysSorted,
    const uint32_t* cellStart,
    const uint32_t* cellEnd,
    sim::DeviceParams dp,
    uint32_t N,
    cudaStream_t s);
extern "C" void LaunchXSPHCompact(
    float4* vel_out,
    const float4* vel_in,
    const float4* pos_pred,
    const uint32_t* indicesSorted,
    const uint32_t* keysSorted,
    const uint32_t* uniqueKeys,
    const uint32_t* offsets,
    const uint32_t* compactCount,
    sim::DeviceParams dp,
    uint32_t N,
    cudaStream_t s);
extern "C" void LaunchXSPHCompactMP(
    float4 * vel_out,
    const float4 * vel_in_fp32, const sim::Half4* vel_in_h4,
    const float4 * pos_pred_fp32, const sim::Half4* pos_pred_h4,
    const uint32_t * indicesSorted, const uint32_t* keysSorted,
    const uint32_t * uniqueKeys, const uint32_t* offsets,
    const uint32_t * compactCount,
    sim::DeviceParams dp, uint32_t N, cudaStream_t s);
extern "C" void LaunchLambdaMP(
    float* lambda,
    const float4 * pos_pred_fp32,
    const sim::Half4 * pos_pred_h4,
    const uint32_t * indicesSorted,
    const uint32_t * keysSorted,
    const uint32_t * cellStart,
    const uint32_t * cellEnd,
    sim::DeviceParams dp,
    uint32_t N,
    cudaStream_t s);
extern "C" void LaunchDeltaApplyMP(
    float4 * pos_pred,
    float4 * delta,
    const float* lambda,
    const float4 * pos_pred_fp32,
    const sim::Half4 * pos_pred_h4,
    const uint32_t * indicesSorted,
    const uint32_t * keysSorted,
    const uint32_t * cellStart,
    const uint32_t * cellEnd,
    sim::DeviceParams dp,
    uint32_t N,
    cudaStream_t s);
extern "C" void LaunchXSPHMP(
    float4 * vel_out,
    const float4 * vel_in_fp32,
    const sim::Half4 * vel_in_h4,
    const float4 * pos_pred_fp32,
    const sim::Half4 * pos_pred_h4,
    const uint32_t * indicesSorted,
    const uint32_t * keysSorted,
    const uint32_t * cellStart,
    const uint32_t * cellEnd,
    sim::DeviceParams dp,
    uint32_t N,
    cudaStream_t s);
//新增：并行压缩段表所需 scratch 预分配函数（在 GraphCapture 前调用）
extern "C" bool EnsureCellCompactScratch(uint32_t N, 
    uint32_t threadsPerBlock /*=256*/);

extern "C" void LaunchVelocity(float4* vel, const float4* pos, const float4* pos_pred, float inv_dt, uint32_t N, cudaStream_t s);
extern "C" void LaunchVelocityMP(float4 * vel, const float4 * pos, const float4 * pos_pred,
    const sim::Half4 * pos_h4, const sim::Half4 * pos_pred_h4,
    float inv_dt, uint32_t N, cudaStream_t s);
extern "C" void LaunchBoundary(float4* pos_pred, float4* vel, sim::GridBounds grid, float restitution, uint32_t N, cudaStream_t s);
extern "C" void LaunchSortPairsQuery(size_t* tempBytes, const uint32_t* d_keys_in, uint32_t* d_keys_out, const uint32_t* d_vals_in, uint32_t* d_vals_out, uint32_t N, cudaStream_t s);
extern "C" void LaunchSortPairs(void* d_temp_storage, size_t tempBytes, uint32_t* d_keys_in, uint32_t* d_keys_out, uint32_t* d_vals_in, uint32_t* d_vals_out, uint32_t N, cudaStream_t s);
extern "C" void LaunchRecycleToNozzleConst(float4* pos, float4* pos_pred, float4* vel, sim::GridBounds grid, float dt, uint32_t N, int enabled, cudaStream_t s);
extern "C" void* GetRecycleKernelPtr();
extern "C" bool LaunchComputeStats(const float4* pos_pred,
    const float4* vel,
    const uint32_t* indices_sorted,
    const uint32_t* cellStart,
    const uint32_t* cellEnd,
    sim::GridBounds grid,
    sim::KernelCoeffs kc,
    float particleMass,
    uint32_t N,
    uint32_t numCells,
    uint32_t sampleStride,
    double* outAvgNeighbors,
    double* outAvgSpeed,
    double* outAvgRhoRel,
    double* outAvgRho,
    cudaStream_t s);
extern "C" bool LaunchComputeStatsBruteforce(const float4* pos_pred,
    const float4* vel,
    sim::KernelCoeffs kc,
    float particleMass,
    uint32_t N,
    uint32_t sampleStride,
    uint32_t maxISamples,
    double* outAvgNeighbors,
    double* outAvgSpeed,
    double* outAvgRhoRel,
    double* outAvgRho,
    cudaStream_t s);

// Poisson-disk 采样辅助
namespace {
    static inline float2 f2_add(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
    static inline float  f2_len2(float2 a) { return a.x * a.x + a.y * a.y; }
    static inline float2 sample_annulus(float rMin, std::mt19937& rng) {
        std::uniform_real_distribution<float> U(0.0f, 1.0f);
        const float R0 = rMin;
        const float R1 = 2.0f * rMin;
        const float u = U(rng);
        const float r = std::sqrt(R0 * R0 + u * (R1 * R1 - R0 * R0));
        const float th = 6.28318530718f * U(rng);
        return make_float2(r * std::cos(th), r * std::sin(th));
    }
    static inline float2 sample_in_disk(float R, std::mt19937& rng) {
        std::uniform_real_distribution<float> U(0.0f, 1.0f);
        const float r = R * std::sqrt(U(rng));
        const float th = 6.28318530718f * U(rng);
        return make_float2(r * std::cos(th), r * std::sin(th));
    }
    static void poisson_disk_in_circle(float R, float rMin, int target,
        std::mt19937& rng,
        std::vector<float2>& outPts) {
        outPts.clear();
        if (R <= 0.0f || rMin <= 0.0f || target <= 0) return;
        const float cell = rMin / 1.41421356237f;
        const float side = 2.0f * R;
        const int gw = (((1) > ((int)std::ceil(side / cell))) ? (1) : ((int)std::ceil(side / cell)));
        const int gh = gw;
        const int nCells = gw * gh;
        std::vector<int> grid(nCells, -1);
        std::vector<int> active;
        active.reserve(target);
        outPts.reserve(target);
        auto toGrid = [&](const float2& p, int& gx, int& gy) {
            const float x = p.x + R;
            const float y = p.y + R;
            gx = std::clamp((int)std::floor(x / cell), 0, gw - 1);
            gy = std::clamp((int)std::floor(y / cell), 0, gh - 1);
            };
        auto gridIdx = [&](int gx, int gy) { return gy * gw + gx; };
        {
            float2 p0 = sample_in_disk(R, rng);
            int gx, gy;
            toGrid(p0, gx, gy);
            grid[gridIdx(gx, gy)] = 0;
            outPts.push_back(p0);
            active.push_back(0);
        }
        std::uniform_int_distribution<int> pick;
        const int kCandidates = 30;
        const float rMin2 = rMin * rMin;
        while (!active.empty() && (int)outPts.size() < target) {
            pick = std::uniform_int_distribution<int>(0, (int)active.size() - 1);
            const int aidx = pick(rng);
            const int pIndex = active[aidx];
            const float2 p = outPts[pIndex];
            bool found = false;
            for (int k = 0; k < kCandidates && (int)outPts.size() < target; ++k) {
                const float2 off = sample_annulus(rMin, rng);
                const float2 q = make_float2(p.x + off.x, p.y + off.y);
                if (f2_len2(q) > R * R) continue;
                int gx, gy;
                toGrid(q, gx, gy);
                bool ok = true;
                for (int yy = (((0) > (gy - 2)) ? (0) : (gy - 2)); yy <= (((gh - 1) < (gy + 2)) ? (gh - 1) : (gy + 2)) && ok; ++yy) {
                    for (int xx = (((0) > (gx - 2)) ? (0) : (gx - 2)); xx <= (((gw - 1) < (gx + 2)) ? (gw - 1) : (gx + 2)) && ok; ++xx) {
                        const int gi = grid[gridIdx(xx, yy)];
                        if (gi < 0) continue;
                        const float2 other = outPts[gi];
                        if (f2_len2(make_float2(other.x - q.x, other.y - q.y)) < rMin2) ok = false;
                    }
                }
                if (!ok) continue;
                const int newIndex = (int)outPts.size();
                outPts.push_back(q);
                active.push_back(newIndex);
                grid[gridIdx(gx, gy)] = newIndex;
                found = true;
            }
            if (!found) {
                active[aidx] = active.back();
                active.pop_back();
            }
        }
    }
} // namespace

namespace sim {
    static inline bool approxEq(float a, float b, float eps = 1e-6f) {
        float da = fabsf(a - b);
        float ma = fmaxf(fabsf(a), fabsf(b));
        return da <= eps * fmaxf(1.0f, ma);
    }
    static inline bool approxEq3(float3 a, float3 b, float eps = 1e-6f) {
        return approxEq(a.x, b.x, eps) && approxEq(a.y, b.y, eps) && approxEq(a.z, b.z, eps);
    }
    static inline bool gridEqual(const GridBounds& a, const GridBounds& b, float eps = 1e-6f) {
        return approxEq3(a.mins, b.mins, eps) && approxEq3(a.maxs, b.maxs, eps) && approxEq(a.cellSize, b.cellSize, eps)
            && (a.dim.x == b.dim.x) && (a.dim.y == b.dim.y) && (a.dim.z == b.dim.z);
    }
    static inline bool kernelEqualRelaxed(const KernelCoeffs& a, const KernelCoeffs& b, float rel = 0.02f) {
        auto nearRel = [&](float x, float y) {
            float mx = fmaxf(fabsf(x), fabsf(y));
            if (mx < 1e-9f) return true;
            return fabsf(x - y) <= rel * mx;
        };
        return nearRel(a.h, b.h) && nearRel(a.inv_h, b.inv_h) && nearRel(a.h2, b.h2)
            && nearRel(a.poly6, b.poly6) && nearRel(a.spiky, b.spiky) && nearRel(a.visc, b.visc);
    }

    static inline bool hostPtrReadable(const void* p) {
#ifdef _WIN32
        MEMORY_BASIC_INFORMATION mbi{};
        if (!p) return false;
        if (VirtualQuery(p, &mbi, sizeof(mbi)) == 0) return false;
        if (mbi.State != MEM_COMMIT) return false;
        if (mbi.Protect & PAGE_NOACCESS) return false;
        if (mbi.Protect & PAGE_GUARD) return false;
        return true;
#else
        return p != nullptr; // 非 Windows 简化处理
#endif
    }
}

namespace sim {

    bool sim::Simulator::initialize(const SimParams& p) {
        prof::Range rInit("Sim.Initialize", prof::Color(0x10, 0x90, 0xF0));
        m_params = p;
        CUDA_CHECK(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));

        const auto& c = console::Instance();
        m_frameTimingEveryN = c.perf.frame_timing_every_n;
        m_frameTimingEnabled = (m_frameTimingEveryN != 0);
        m_useHashedGrid = c.perf.use_hashed_grid;

        uint32_t capacity = (p.maxParticles > 0) ? p.maxParticles : p.numParticles;
        if (capacity == 0) capacity = 1;

        bool needHalf = (p.precision.positionStore == sim::NumericType::FP16_Packed) ||
            (p.precision.velocityStore == sim::NumericType::FP16_Packed) ||
            (p.precision.predictedPosStore == sim::NumericType::FP16_Packed);
        if (needHalf) {
            m_bufs.allocateWithPrecision(p.precision, capacity);
        }
        else {
            m_bufs.allocate(capacity);
        }
        UpdateDevicePrecisionView(m_bufs, p.precision);

        m_bufs.ensureCompactCapacity(capacity);
        if (!buildGrid(m_params)) return false;

        if (capacity > 0) {
            std::vector<uint32_t> h_idx(capacity);
            for (uint32_t i = 0; i < capacity; ++i) h_idx[i] = i;
            CUDA_CHECK(cudaMemcpy(m_bufs.d_indices, h_idx.data(), sizeof(uint32_t) * capacity, cudaMemcpyHostToDevice));
        }
        if (p.numParticles > 0) {
            CUDA_CHECK(cudaMemcpy(m_bufs.d_pos_pred, m_bufs.d_pos,
                sizeof(float4) * p.numParticles, cudaMemcpyDeviceToDevice));
            if (needHalf) {
                m_bufs.packAllToHalf(p.numParticles, m_stream);
            }
        }

        // ===== 半精镜像模式一次性日志（M1） =====
        if (!m_precisionLogged) {
            const auto& pr = m_params.precision;
            bool posHalf = (pr.positionStore == sim::NumericType::FP16_Packed);
            bool velHalf = (pr.velocityStore == sim::NumericType::FP16_Packed);
            bool predHalf = (pr.predictedPosStore == sim::NumericType::FP16_Packed);
            std::fprintf(stderr,
                "[Precision] Init | posHalf=%d velHalf=%d predHalf=%d | core=%u fp32Acc=%d halfIntr=%d\n",
                posHalf, velHalf, predHalf,
                (unsigned)pr.coreCompute, pr.forceFp32Accumulate ? 1 : 0, pr.enableHalfIntrinsics ? 1 : 0);

            if ((posHalf && !m_bufs.d_pos_h4) ||
                (velHalf && !m_bufs.d_vel_h4) ||
                (predHalf && !m_bufs.d_pos_pred_h4)) {
                std::fprintf(stderr,
                    "[Precision][Warn] Requested half mirror not allocated (pos=%p vel=%p pred=%p).\n",
                    (void*)m_bufs.d_pos_h4, (void*)m_bufs.d_vel_h4, (void*)m_bufs.d_pos_pred_h4);
            }
            else if (posHalf || velHalf || predHalf) {
                size_t extraBytes = 0;
                if (posHalf)  extraBytes += sizeof(sim::Half4) * m_bufs.capacity;
                if (velHalf)  extraBytes += sizeof(sim::Half4) * m_bufs.capacity;
                if (predHalf) extraBytes += sizeof(sim::Half4) * m_bufs.capacity;
                std::fprintf(stderr,
                    "[Precision] Mirror allocated capacity=%u overhead=%.2f MB\n",
                    m_bufs.capacity, extraBytes / (1024.0 * 1024.0));
            }
            bool autoMapped = (m_params.useMixedPrecision &&
                posHalf && velHalf && predHalf &&
                pr.coreCompute == sim::NumericType::FP32 &&
                pr.forceFp32Accumulate);
            std::fprintf(stderr, "[Precision] autoMapMixed=%d\n", autoMapped ? 1 : 0);
            m_precisionLogged = true;
        }

        m_canPingPongPos = true;
        m_graphDirty = true;
        m_captured = {};
        m_cachedNodesReady = false;
        m_nodeRecycleFull = m_nodeRecycleCheap = nullptr;
        m_lastFrameMs = -1.0f;
        m_evCursor = 0;
        m_lastParamUpdateFrame = -1;
        return true;
    }

    void Simulator::shutdown() {
        if (m_extPosPred) {
            cudaDestroyExternalMemory(m_extPosPred);
            m_extPosPred = nullptr;
            m_bufs.detachExternalPosPred();
        }
        if (m_graphExecFull) { cudaGraphExecDestroy(m_graphExecFull);  m_graphExecFull = nullptr; }
        if (m_graphFull) { cudaGraphDestroy(m_graphFull);          m_graphFull = nullptr; }
        if (m_graphExecCheap) { cudaGraphExecDestroy(m_graphExecCheap); m_graphExecCheap = nullptr; }
        if (m_graphCheap) { cudaGraphDestroy(m_graphCheap);         m_graphCheap = nullptr; }
        for (int i = 0; i < 2; ++i) {
            if (m_evStart[i]) { cudaEventDestroy(m_evStart[i]); m_evStart[i] = nullptr; }
            if (m_evEnd[i]) { cudaEventDestroy(m_evEnd[i]); m_evEnd[i] = nullptr; }
        }
        if (m_stream) { cudaStreamDestroy(m_stream); m_stream = nullptr; }
    }

    bool Simulator::buildGrid(const SimParams& p) {
        int3 dim;
        dim.x = int(ceilf((p.grid.maxs.x - p.grid.mins.x) / p.grid.cellSize));
        dim.y = int(ceilf((p.grid.maxs.y - p.grid.mins.y) / p.grid.cellSize));
        dim.z = int(ceilf((p.grid.maxs.z - p.grid.mins.z) / p.grid.cellSize));
        m_numCells = uint32_t(dim.x) * uint32_t(dim.y) * uint32_t(dim.z);
        if (m_numCells == 0) return false;
        m_bufs.allocGridRanges(m_numCells);
        m_params.grid = p.grid;
        m_params.grid.dim = dim;
        return true;
    }

    bool Simulator::updateGridIfNeeded(const SimParams& p) {
        int3 dim;
        dim.x = int(ceilf((p.grid.maxs.x - p.grid.mins.x) / p.grid.cellSize));
        dim.y = int(ceilf((p.grid.maxs.y - p.grid.mins.y) / p.grid.cellSize));
        dim.z = int(ceilf((p.grid.maxs.z - p.grid.mins.z) / p.grid.cellSize));
        uint32_t newNumCells = uint32_t(dim.x) * uint32_t(dim.y) * uint32_t(dim.z);
        bool gridGeomChanged = false;
        if (newNumCells != m_numCells) gridGeomChanged = true;
        if (!gridEqual(p.grid, m_captured.grid)) gridGeomChanged = true;
        if (gridGeomChanged) {
            m_bufs.resizeGridRanges(newNumCells);
            m_numCells = newNumCells;
            m_graphDirty = true;
        }
        m_params.grid = p.grid;
        m_params.grid.dim = dim;
        m_bufs.ensureCompactCapacity(p.maxParticles > 0 ? p.maxParticles : p.numParticles);
        return gridGeomChanged;
    }

    bool Simulator::ensureSortTemp(std::size_t bytes) {
        m_bufs.ensureSortTemp(bytes);
        return true;
    }

    bool Simulator::cacheGraphNodes() {
        m_nodeRecycleFull = nullptr;
        m_nodeRecycleCheap = nullptr;
        m_kpRecycleBaseFull = {};
        m_kpRecycleBaseCheap = {};
        m_cachedNodesReady = false;

        // 速度相关节点缓存清理
        m_velNodesFull.clear();
        m_velNodesCheap.clear();
        m_velNodeParamsBaseFull.clear();
        m_velNodeParamsBaseCheap.clear();
        m_cachedVelNodes = false;

        void* targetRecycle = GetRecycleKernelPtr();

        auto scanGraph = [&](cudaGraph_t g,
            std::vector<cudaGraphNode_t>& outVelNodes,
            std::vector<cudaKernelNodeParams>& outVelParams,
            std::vector<cudaGraphNode_t>& outPosNodes,
            std::vector<cudaKernelNodeParams>& outPosParams,
            bool recordRecycle) {
                if (!g) return;
                size_t numNodes = 0;
                CUDA_CHECK(cudaGraphGetNodes(g, nullptr, &numNodes));
                if (numNodes == 0) return;
                std::vector<cudaGraphNode_t> nodes(numNodes);
                CUDA_CHECK(cudaGraphGetNodes(g, nodes.data(), &numNodes));
                for (auto n : nodes) {
                    cudaGraphNodeType t;
                    CUDA_CHECK(cudaGraphNodeGetType(n, &t));
                    if (t != cudaGraphNodeTypeKernel) continue;
                    cudaKernelNodeParams kp{};
                    CUDA_CHECK(cudaGraphKernelNodeGetParams(n, &kp));

                    if (recordRecycle && kp.func == targetRecycle) {
                        m_nodeRecycleFull = n;
                        m_kpRecycleBaseFull = kp;
                    }
                    else if (!recordRecycle && kp.func == targetRecycle) {
                        m_nodeRecycleCheap = n;
                        m_kpRecycleBaseCheap = kp;
                    }

                    if (!kp.kernelParams) continue;
                    void** params = (void**)kp.kernelParams;

                    // 监控速度指针
                    std::unordered_set<const void*> watchedVel{
                        (const void*)m_bufs.d_vel,
                        (const void*)m_bufs.d_delta
                    };
                    // 监控位置指针
                    std::unordered_set<const void*> watchedPos{
                        (const void*)m_bufs.d_pos,
                        (const void*)m_bufs.d_pos_pred
                    };

                    bool hasVel = false;
                    bool hasPos = false;
                    const int kMaxScan = 32;
                    for (int i = 0; i < kMaxScan; ++i) {
                        void* argSlot = params[i];
                        if (!argSlot) break;
                        if (!hostPtrReadable(argSlot)) break;
                        void* val = *(void**)argSlot;
                        if (watchedVel.count(val)) hasVel = true;
                        if (watchedPos.count(val)) hasPos = true;
                        if (hasVel && hasPos) break;
                    }
                    if (hasVel) {
                        outVelNodes.push_back(n);
                        outVelParams.push_back(kp);
                    }
                    if (hasPos) {
                        outPosNodes.push_back(n);
                        outPosParams.push_back(kp);
                    }
                }
            };

        scanGraph(m_graphFull,
            m_velNodesFull, m_velNodeParamsBaseFull,
            m_posNodesFull, m_posNodeParamsBaseFull,
            true);
        scanGraph(m_graphCheap,
            m_velNodesCheap, m_velNodeParamsBaseCheap,
            m_posNodesCheap, m_posNodeParamsBaseCheap,
            false);

        m_cachedVelNodes = true;
        m_cachedPosNodes = true;
        m_cachedNodesReady = true;
        return true;
    }

    bool Simulator::structuralGraphChange(const SimParams& p) const {
        if (!m_graphExecFull || !m_graphExecCheap) return true;
        if (p.solverIters != m_captured.solverIters) return true;
        if (p.maxNeighbors != m_captured.maxNeighbors) return true;
        if (p.numParticles != m_captured.numParticles) return true;
        int3 dim;
        dim.x = int(ceilf((p.grid.maxs.x - p.grid.mins.x) / p.grid.cellSize));
        dim.y = int(ceilf((p.grid.maxs.y - p.grid.mins.y) / p.grid.cellSize));
        dim.z = int(ceilf((p.grid.maxs.z - p.grid.mins.z) / p.grid.cellSize));
        uint32_t newNumCells = uint32_t(dim.x) * uint32_t(dim.y) * uint32_t(dim.z);
        if (newNumCells != m_captured.numCells) return true;
        if (!gridEqual(p.grid, m_captured.grid)) return true;
        return false;
    }

    bool Simulator::paramOnlyGraphChange(const SimParams& p) const {
        if (structuralGraphChange(p)) return false;
        // 阈值：若变化极小则忽略避免频繁 update
        const float dtRel = fabsf(p.dt - m_captured.dt) / fmaxf(1e-9f, fmaxf(p.dt, m_captured.dt));
        if (dtRel < 0.002f && approxEq3(p.gravity, m_captured.gravity) && approxEq(p.restDensity, m_captured.restDensity) && kernelEqualRelaxed(p.kernel, m_captured.kernel)) {
            return false; // 微小变化忽略
        }
        if (!approxEq(p.dt, m_captured.dt)) return true;
        if (!approxEq3(p.gravity, m_captured.gravity)) return true;
        if (!approxEq(p.restDensity, m_captured.restDensity)) return true;
        if (!kernelEqualRelaxed(p.kernel, m_captured.kernel)) return true;
        return false;
    }

    bool Simulator::captureGraphIfNeeded(const SimParams& p) {
        if (!m_graphDirty) return true;
        if (!m_graphPointersChecked) {
            if (!m_bufs.d_pos || !m_bufs.d_vel || !m_bufs.d_pos_pred) {
                std::fprintf(stderr,
                    "[Graph][Error] Required buffers null before capture (pos=%p vel=%p pos_pred=%p)\n",
                    (void*)m_bufs.d_pos, (void*)m_bufs.d_vel, (void*)m_bufs.d_pos_pred);
            }
            m_graphPointersChecked = true;
        }
        prof::Range rCapture("Sim.GraphCapture", prof::Color(0xE0, 0x60, 0x20));
        {
            size_t tempBytes = 0;
            ::LaunchSortPairsQuery(&tempBytes,
                m_bufs.d_cellKeys, m_bufs.d_cellKeys_sorted,
                m_bufs.d_indices, m_bufs.d_indices_sorted,
                p.numParticles, m_stream);
            ensureSortTemp(tempBytes);
        }

        // 新增：若使用 hashed/紧凑网格，提前确保并行段构建的 scratch（避免 GraphCapture 内 cudaMalloc）
        if (m_useHashedGrid) {
            if (!EnsureCellCompactScratch(p.numParticles, 256)) {
                if (console::Instance().debug.printErrors) {
                    std::fprintf(stderr, "[Error] EnsureCellCompactScratch failed (GraphCapture).\n");
                }
                return false;
            }
        }

        if (m_graphExecFull) { cudaGraphExecDestroy(m_graphExecFull);  m_graphExecFull = nullptr; }
        if (m_graphFull) { cudaGraphDestroy(m_graphFull);          m_graphFull = nullptr; }
        if (m_graphExecCheap) { cudaGraphExecDestroy(m_graphExecCheap); m_graphExecCheap = nullptr; }
        if (m_graphCheap) { cudaGraphDestroy(m_graphCheap);         m_graphCheap = nullptr; }
        CUDA_CHECK(cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeGlobal));
        kIntegratePred(m_stream, p);
        kHashKeys(m_stream, p);
        kSort(m_stream, p);
        if (m_useHashedGrid) kCellRangesCompact(m_stream, p);
        else                 kCellRanges(m_stream, p);
        for (int i = 0; i < p.solverIters; ++i) kSolveIter(m_stream, p);
        kVelocityAndPost(m_stream, p);
        CUDA_CHECK(cudaStreamEndCapture(m_stream, &m_graphFull));
        CUDA_CHECK(cudaGraphInstantiate(&m_graphExecFull, m_graphFull, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphUpload(m_graphExecFull, m_stream));
        CUDA_CHECK(cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeGlobal));
        kIntegratePred(m_stream, p);
        for (int i = 0; i < p.solverIters; ++i) kSolveIter(m_stream, p);
        kVelocityAndPost(m_stream, p);
        CUDA_CHECK(cudaStreamEndCapture(m_stream, &m_graphCheap));
        CUDA_CHECK(cudaGraphInstantiate(&m_graphExecCheap, m_graphCheap, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphUpload(m_graphExecCheap, m_stream));
        cacheGraphNodes();
        int3 dim;
        dim.x = int(ceilf((p.grid.maxs.x - p.grid.mins.x) / p.grid.cellSize));
        dim.y = int(ceilf((p.grid.maxs.y - p.grid.mins.y) / p.grid.cellSize));
        dim.z = int(ceilf((p.grid.maxs.z - p.grid.mins.z) / p.grid.cellSize));
        uint32_t newNumCells = uint32_t(dim.x) * uint32_t(dim.y) * uint32_t(dim.z);
        m_captured.numParticles = p.numParticles;
        m_captured.numCells = newNumCells;
        m_captured.solverIters = p.solverIters;
        m_captured.maxNeighbors = p.maxNeighbors;
        m_captured.sortEveryN = p.sortEveryN;
        m_captured.grid = p.grid;
        m_captured.kernel = p.kernel;
        m_captured.dt = p.dt;
        m_captured.gravity = p.gravity;
        m_captured.restDensity = p.restDensity;
        m_graphDirty = false;
        m_paramDirty = false;
        return true;
    }

    bool Simulator::updateGraphsParams(const SimParams& p) {
        if (!m_paramDirty) return true;
        if (!m_bufs.d_pos || !m_bufs.d_vel || !m_bufs.d_pos_pred) {
            std::fprintf(stderr,
                "[GraphUpdate][Error] Buffer handle invalid (pos=%p vel=%p pred=%p)\n",
                (void*)m_bufs.d_pos, (void*)m_bufs.d_vel, (void*)m_bufs.d_pos_pred);
        }
        if (!m_graphExecFull || !m_graphExecCheap) return false;
        const auto& c = console::Instance();
        const int minInterval = c.perf.graph_param_update_min_interval;
        if (minInterval > 0 && m_lastParamUpdateFrame >= 0 && (m_frameIndex - m_lastParamUpdateFrame) < minInterval) {
            m_paramDirty = false;
            return true;
        }
        prof::Range rUpd("Sim.GraphUpdate", prof::Color(0xA0, 0x50, 0x10));

        if (m_useHashedGrid) {
            if (!EnsureCellCompactScratch(p.numParticles, 256)) {
                if (c.debug.printErrors) {
                    std::fprintf(stderr, "[Error] EnsureCellCompactScratch failed (GraphUpdate).\n");
                }
                m_graphDirty = true;
                return captureGraphIfNeeded(p);
            }
        }

        cudaGraph_t newFull = nullptr;
        cudaGraph_t newCheap = nullptr;
        CUDA_CHECK(cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeGlobal));
        kIntegratePred(m_stream, p);
        kHashKeys(m_stream, p);
        kSort(m_stream, p);
        if (m_useHashedGrid) kCellRangesCompact(m_stream, p); else kCellRanges(m_stream, p);
        for (int i = 0; i < p.solverIters; ++i) kSolveIter(m_stream, p);
        kVelocityAndPost(m_stream, p);
        CUDA_CHECK(cudaStreamEndCapture(m_stream, &newFull));

        CUDA_CHECK(cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeGlobal));
        kIntegratePred(m_stream, p);
        for (int i = 0; i < p.solverIters; ++i) kSolveIter(m_stream, p);
        kVelocityAndPost(m_stream, p);
        CUDA_CHECK(cudaStreamEndCapture(m_stream, &newCheap));

        cudaGraphExecUpdateResultInfo updateResultFull{};
        cudaError_t eFull = cudaGraphExecUpdate(m_graphExecFull, newFull, &updateResultFull);
        cudaGraphExecUpdateResultInfo updateResultCheap{};
        cudaError_t eCheap = cudaGraphExecUpdate(m_graphExecCheap, newCheap, &updateResultCheap);

        bool needRebuild = false;
        if (eFull != cudaSuccess || updateResultFull.result != cudaGraphExecUpdateSuccess)  needRebuild = true;
        if (eCheap != cudaSuccess || updateResultCheap.result != cudaGraphExecUpdateSuccess) needRebuild = true;

        if (newFull)  cudaGraphDestroy(newFull);
        if (newCheap) cudaGraphDestroy(newCheap);

        if (needRebuild) {
            m_graphDirty = true;
            return captureGraphIfNeeded(p); // captureGraphIfNeeded 内部已调用 cacheGraphNodes()
        }

        // ===== 新增：成功更新后刷新节点缓存，避免使用失效 node 句柄 =====
        cacheGraphNodes();

        m_captured.dt = p.dt;
        m_captured.gravity = p.gravity;
        m_captured.restDensity = p.restDensity;
        m_captured.kernel = p.kernel;
        m_paramDirty = false;
        m_lastParamUpdateFrame = m_frameIndex;
        return true;
    }

    static inline float3 normalize3(float3 v) {
        float n2 = v.x * v.x + v.y * v.y + v.z * v.z;
        if (n2 <= 1e-20f) return make_float3(0, -1, 0);
        float invn = 1.0f / sqrtf(n2);
        return make_float3(v.x * invn, v.y * invn, v.z * invn);
    }
    static inline float3 cross3(float3 a, float3 b) {
        return make_float3(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        );
    }
    static inline float3 add3(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
    static inline float3 mul3(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
    bool Simulator::step(const SimParams& p) {
        prof::Range rFrame("Sim.Step", prof::Color(0x30, 0x30, 0xA0));
        m_params = p;

        // 动态热更新采样频率（允许热重载调节）
        {
            const auto& cHot = console::Instance();
            if (m_frameTimingEveryN != cHot.perf.frame_timing_every_n) {
                m_frameTimingEveryN = cHot.perf.frame_timing_every_n;
                m_frameTimingEnabled = (m_frameTimingEveryN != 0);
                // 若关闭后，不再使用已存在事件（不销毁以简化；下次 initialize 会重建）
            }
        }
        {
            prof::Range rEmit("EmitParticles", prof::Color(0xFF, 0x90, 0x30));
            int3 dim;
            dim.x = int(ceilf((m_params.grid.maxs.x - m_params.grid.mins.x) / m_params.grid.cellSize));
            dim.y = int(ceilf((m_params.grid.maxs.y - m_params.grid.mins.y) / m_params.grid.cellSize));
            dim.z = int(ceilf((m_params.grid.maxs.z - m_params.grid.mins.z) / m_params.grid.cellSize));
            m_params.grid.dim = dim;
        }
        const auto& c = console::Instance();
        sim::EmitParams ep{};
        console::BuildEmitParams(c, ep);
        const bool faucetFillEnable = c.sim.faucetFillEnable;
        const uint32_t emitPerStep = c.sim.emitPerStep;
        const bool recycleToNozzle = c.sim.recycleToNozzle;

        {
            const uint32_t capacity = m_bufs.capacity;
            if (m_params.maxParticles == 0) m_params.maxParticles = capacity;
            m_params.maxParticles = std::min<uint32_t>(m_params.maxParticles, capacity);

            if (faucetFillEnable && m_params.numParticles < m_params.maxParticles) {
                const uint32_t canEmit = m_params.maxParticles - m_params.numParticles;
                const uint32_t emit = std::min<uint32_t>(emitPerStep, canEmit);

                if (emit > 0) {
                    const uint32_t begin = m_params.numParticles;
                    const uint32_t end = begin + emit;

                    static float4* s_h_pos = nullptr;
                    static float4* s_h_vel = nullptr;
                    static uint32_t s_h_cap = 0;
                    if (s_h_cap < emit) {
                        if (s_h_pos) { cudaFreeHost(s_h_pos); s_h_pos = nullptr; }
                        if (s_h_vel) { cudaFreeHost(s_h_vel); s_h_vel = nullptr; }
                        CUDA_CHECK(cudaMallocHost((void**)&s_h_pos, sizeof(float4) * emit));
                        CUDA_CHECK(cudaMallocHost((void**)&s_h_vel, sizeof(float4) * emit));
                        s_h_cap = emit;
                    }
                    float4* h_pos = s_h_pos;
                    float4* h_vel = s_h_vel;

                    const float3 dir = normalize3(ep.nozzleDir);
                    float3 ref = fabsf(dir.y) < 0.99f ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
                    float3 u = normalize3(cross3(ref, dir));
                    float3 v = cross3(dir, u);

                    std::mt19937 rng(0xBADC0DEu ^ (uint32_t)m_frameIndex ^ begin);
                    std::uniform_real_distribution<float> U(0.0f, 1.0f);

                    const float h = m_params.kernel.h;
                    const float R = (ep.nozzleRadius > 0.f) ? ep.nozzleRadius : (h * 1.5f);
                    const float minSpacing = (((1e-6f) > (c.sim.poisson_min_spacing_factor_h * h)) ? (1e-6f) : (c.sim.poisson_min_spacing_factor_h * h));

                    std::vector<float2> pts;
                    poisson_disk_in_circle(R, minSpacing, (int)emit, rng, pts);

                    const uint32_t nPoisson = (uint32_t)pts.size();
                    const uint32_t nRemain = emit - nPoisson;

                    for (uint32_t i = 0; i < nPoisson; ++i) {
                        const float2 xy = pts[i];
                        float3 offset = add3(mul3(u, xy.x), mul3(v, xy.y));
                        float3 p0 = add3(ep.nozzlePos, offset);
                        float3 vel0 = mul3(dir, (ep.nozzleSpeed > 0.f ? ep.nozzleSpeed : 1.0f));
                        h_pos[i] = make_float4(p0.x, p0.y, p0.z, 1.0f);
                        h_vel[i] = make_float4(vel0.x, vel0.y, vel0.z, 0.0f);
                    }

                    for (uint32_t i = 0; i < nRemain; ++i) {
                        const float u0 = U(rng);
                        const float u1 = U(rng);
                        const float r = R * sqrtf(u0);
                        const float th = 6.28318530718f * u1;
                        float3 offset = add3(mul3(u, r * cosf(th)), mul3(v, r * sinf(th)));
                        float3 p0 = add3(ep.nozzlePos, offset);
                        float3 vel0 = mul3(dir, (ep.nozzleSpeed > 0.f ? ep.nozzleSpeed : 1.0f));
                        const uint32_t idx = nPoisson + i;
                        h_pos[idx] = make_float4(p0.x, p0.y, p0.z, 1.0f);
                        h_vel[idx] = make_float4(vel0.x, vel0.y, vel0.z, 0.0f);
                    }

                    if (c.sim.emit_jitter_enable) {
                        const float h_use = (m_params.kernel.h > 0.f) ? m_params.kernel.h : (R > 0.f ? R : 1.f);
                        const float amp = c.sim.emit_jitter_scale_h * h_use;
                        if (amp > 0.f) {
                            std::mt19937 jrng(c.sim.emit_jitter_seed ^ (uint32_t)m_frameIndex);
                            std::uniform_real_distribution<float> J(-1.0f, 1.0f);
                            for (uint32_t i = 0; i < emit; ++i) {
                                float ox, oy, oz;
                                for (;;) {
                                    ox = J(jrng); oy = J(jrng); oz = J(jrng);
                                    if (ox * ox + oy * oy + oz * oz <= 1.0f) break;
                                }
                                ox *= amp; oy *= amp; oz *= amp;
                                float4 p = h_pos[i];
                                p.x += ox; p.y += oy; p.z += oz;
                                // 钳制回域内
                                p.x = fminf(fmaxf(p.x, m_params.grid.mins.x), m_params.grid.maxs.x);
                                p.y = fminf(fmaxf(p.y, m_params.grid.mins.y), m_params.grid.maxs.y);
                                p.z = fminf(fmaxf(p.z, m_params.grid.mins.z), m_params.grid.maxs.z);
                                h_pos[i] = p;
                            }
                        }
                    }

                    CUDA_CHECK(cudaMemcpyAsync(m_bufs.d_pos + begin, h_pos,
                        sizeof(float4)* emit, cudaMemcpyHostToDevice, m_stream));
                    // 保持速度写
                    CUDA_CHECK(cudaMemcpyAsync(m_bufs.d_vel + begin, h_vel,
                        sizeof(float4)* emit, cudaMemcpyHostToDevice, m_stream));

                    m_params.numParticles = end;
                }
            }
            m_bufs.packAllToHalf(m_params.numParticles, m_stream);
        }

        /// 若仍不足容量则扩容（兜底）
        if (m_params.numParticles > m_bufs.capacity) {
            const auto& pr = m_params.precision;
            bool needHalf = (pr.positionStore == sim::NumericType::FP16_Packed ||
                pr.positionStore == sim::NumericType::FP16) ||
                (pr.velocityStore == sim::NumericType::FP16_Packed ||
                    pr.velocityStore == sim::NumericType::FP16) ||
                (pr.predictedPosStore == sim::NumericType::FP16_Packed ||
                    pr.predictedPosStore == sim::NumericType::FP16);
            if (needHalf) m_bufs.allocateWithPrecision(pr, m_params.numParticles);
            else          m_bufs.allocate(m_params.numParticles);
            UpdateDevicePrecisionView(m_bufs, m_params.precision);
            // 重新写索引...
            if (needHalf) {
                std::fprintf(stderr,
                    "[Precision] Reallocate for expansion newCap=%u | pos_h4=%p vel_h4=%p pred_h4=%p\n",
                    m_bufs.capacity,
                    (void*)m_bufs.d_pos_h4,
                    (void*)m_bufs.d_vel_h4,
                    (void*)m_bufs.d_pos_pred_h4);
            }
            m_graphDirty = true;
        }

        // 每帧：更新设备常量（喷口/回收参数）
        {
            SetEmitParamsAsync(&ep, m_stream);
        }

        // 更新网格（可能触发结构性重建）
        updateGridIfNeeded(m_params);

        // 结构性判定
        if (structuralGraphChange(m_params)) {
            m_graphDirty = true;
        }
        else if (paramOnlyGraphChange(m_params)) {
            m_paramDirty = true;
        }

        // 先处理结构性重建，再处理参数更新
        if (m_graphDirty) {
            captureGraphIfNeeded(m_params);
        }
        else if (m_paramDirty) {
            updateGraphsParams(m_params);
        }

        // 选择 Full / Cheap
        const int everyN = (c.perf.sort_compact_every_n <= 0) ? 1 : c.perf.sort_compact_every_n;
        const bool needFullThisFrame =
            (m_frameIndex == 0) ||
            (m_lastFullFrame < 0) ||
            ((m_frameIndex - m_lastFullFrame) >= everyN) ||
            (m_params.numParticles != m_captured.numParticles); // 若粒子数真正变化（结构性），已重建

        prof::Mark(needFullThisFrame ? "Launch.FullGraph" : "Launch.CheapGraph",
            needFullThisFrame ? prof::Color(0xD0, 0x40, 0x20) : prof::Color(0x20, 0xA0, 0x40));

        if (!m_graphExecFull || !m_graphExecCheap) {
            if (c.debug.printErrors) {
                std::fprintf(stderr, "Simulator::step: CUDA Graph not ready (exec == nullptr)\n");
            }
            return false;
        }
        const bool doTiming = m_frameTimingEnabled &&
            (m_frameTimingEveryN <= 1 || (m_frameIndex % m_frameTimingEveryN) == 0);

        int cur = (m_evCursor & 1);
        int prev = ((m_evCursor + 1) & 1);
        if (doTiming && m_frameTimingEnabled) {
            // ---- Lazy create timing events (Fix invalid resource handle) ----
            if (!m_evStart[0] || !m_evStart[1] || !m_evEnd[0] || !m_evEnd[1]) {
                for (int i = 0; i < 2; ++i) {
                    if (!m_evStart[i]) {
                        cudaError_t ce = cudaEventCreate(&m_evStart[i]);
                        if (ce != cudaSuccess) {
                            std::fprintf(stderr, "[Timing][Error] cudaEventCreate start[%d] failed: %s\n",
                                i, cudaGetErrorString(ce));
                        }
                    }
                    if (!m_evEnd[i]) {
                        cudaError_t ce = cudaEventCreate(&m_evEnd[i]);
                        if (ce != cudaSuccess) {
                            std::fprintf(stderr, "[Timing][Error] cudaEventCreate end[%d] failed: %s\n",
                                i, cudaGetErrorString(ce));
                        }
                    }
                }
            }
            // 若仍有事件未成功创建，则跳过计时避免继续报错
            if (m_evStart[cur] && m_evEnd[cur]) {
                CUDA_CHECK(cudaEventRecord(m_evStart[cur], m_stream));
            }
        }

        auto updateRecycleGridDim = [&](cudaGraphExec_t exec,
            cudaGraphNode_t& node,
            cudaKernelNodeParams& base) {
                if (!exec || !node || m_params.numParticles == 0) return;
                uint32_t blocks = (m_params.numParticles + 255u) / 256u;
                if (blocks == 0u) blocks = 1u;
                cudaKernelNodeParams kp = base;
                kp.gridDim = dim3(blocks, 1, 1);
                cudaError_t err = cudaGraphExecKernelNodeSetParams(exec, node, &kp);
                if (err == cudaErrorInvalidResourceHandle) {
                    std::fprintf(stderr,
                        "[Graph][Warn] Kernel node invalid, recache & retry (node=%p, N=%u)\n",
                        (void*)node, m_params.numParticles);
                    // 重新扫描当前图
                    cacheGraphNodes();
                    // 根据当前执行路径重新选择 node/base
                    if (exec == m_graphExecFull) {
                        node = m_nodeRecycleFull;
                        base = m_kpRecycleBaseFull;
                    }
                    else if (exec == m_graphExecCheap) {
                        node = m_nodeRecycleCheap;
                        base = m_kpRecycleBaseCheap;
                    }
                    if (node) {
                        kp = base;
                        kp.gridDim = dim3(blocks, 1, 1);
                        cudaError_t err2 = cudaGraphExecKernelNodeSetParams(exec, node, &kp);
                        if (err2 != cudaSuccess) {
                            std::fprintf(stderr,
                                "[Graph][Error] Retry setParams failed err=%d (%s)\n",
                                (int)err2, cudaGetErrorString(err2));
                        }
                    }
                    else {
                        std::fprintf(stderr, "[Graph][Error] Recycle node not found after recache.\n");
                    }
                }
            };

        if (needFullThisFrame) {
            updateRecycleGridDim(m_graphExecFull, m_nodeRecycleFull, m_kpRecycleBaseFull);
        }
        else {
            updateRecycleGridDim(m_graphExecCheap, m_nodeRecycleCheap, m_kpRecycleBaseCheap);
        }

        if (c.perf.use_cuda_graphs) {
            if (needFullThisFrame) {
                CUDA_CHECK(cudaGraphLaunch(m_graphExecFull, m_stream));
                m_lastFullFrame = m_frameIndex;
            }
            else {
                CUDA_CHECK(cudaGraphLaunch(m_graphExecCheap, m_stream));
            }
        }
        else {
            kIntegratePred(m_stream, m_params);
            if (needFullThisFrame) {
                kHashKeys(m_stream, m_params);
                kSort(m_stream, m_params);
                if (m_useHashedGrid) kCellRangesCompact(m_stream, m_params);
                else                 kCellRanges(m_stream, m_params);
                m_lastFullFrame = m_frameIndex;
            }
            for (int i = 0; i < m_params.solverIters; ++i) kSolveIter(m_stream, m_params);
            kVelocityAndPost(m_stream, m_params);
        }

        if (doTiming && m_frameTimingEnabled) {
            if (m_evEnd[cur]) {
                CUDA_CHECK(cudaEventRecord(m_evEnd[cur], m_stream));
            }
            if (m_evCursor > 0 && m_evStart[prev] && m_evEnd[prev]) {
                const cudaError_t q = cudaEventQuery(m_evEnd[prev]);
                if (q == cudaSuccess) {
                    float ms = 0.f;
                    CUDA_CHECK(cudaEventElapsedTime(&ms, m_evStart[prev], m_evEnd[prev]));
                    m_lastFrameMs = ms;
                }
            }
            ++m_evCursor;
        }

        // 可选：输出网格压缩统计（会产生 D2H）
        if (m_useHashedGrid && c.perf.log_grid_compact_stats) {
            uint32_t M = 0;
            CUDA_CHECK(cudaMemcpy((void*)&M, (const void*)m_bufs.d_compactCount, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            if (M > 0u) {
                const double avgOcc = double(m_params.numParticles) / double(M);
                std::fprintf(stderr, "[GridSparse] Frame=%d | N=%u | M=%u | avgOcc=%.2f\n",
                    m_frameIndex, m_params.numParticles, M, avgOcc);
            } else {
                std::fprintf(stderr, "[GridSparse] Frame=%d | N=%u | M=0\n", m_frameIndex, m_params.numParticles);
            }
        }

        // 可选：数值稳定性日志（注意：此路径会触发同步/D2H，建议关闭以追求极致性能）
        if (c.debug.logStabilityBasic && c.debug.logEveryN > 0 && m_params.numParticles > 0) {
            prof::Range rStab("Stability.Log", prof::Color(0x90, 0x50, 0x30));
            const int logEveryN = (((1) > (c.debug.logEveryN)) ? (1) : (c.debug.logEveryN));
            if ((m_frameIndex % (uint64_t)logEveryN) == 0ull) {
                double avgN = 0.0, avgV = 0.0, avgRhoRel = 0.0, avgR = 0.0;
                const uint32_t stride = (c.debug.logSampleStride <= 0 ? 1u : (uint32_t)c.debug.logSampleStride);
                (void)LaunchComputeStats(
                    m_bufs.d_pos_pred,
                    m_bufs.d_vel,
                    m_bufs.d_indices_sorted,
                    m_bufs.d_cellStart,
                    m_bufs.d_cellEnd,
                    m_params.grid,
                    m_params.kernel,
                    m_params.particleMass,
                    m_params.numParticles,
                    m_numCells,
                    stride,
                    &avgN, &avgV, &avgRhoRel, &avgR,
                    m_stream);
                CUDA_CHECK(cudaStreamSynchronize(m_stream)); // 若追求极限性能，请关闭 logStabilityBasic 以跳过这条同步

                static double s_prevAvgV = -1.0;
                const double v_decay = (s_prevAvgV > 0.0) ? (avgV / (((1e-12) > (s_prevAvgV)) ? (1e-12) : (s_prevAvgV))) : 0.0;
                s_prevAvgV = avgV;

                double cfl_est_avg = 0.0;
                if (c.debug.logCFL && m_params.kernel.h > 0.0f) {
                    cfl_est_avg = (double)m_params.dt * avgV / (double)m_params.kernel.h;
                }

                double lamMin = 0.0, lamMax = 0.0, lamAvg = 0.0;
                if (c.debug.logLambda) {
                    const uint32_t M = std::min<uint32_t>(m_params.numParticles, (uint32_t)((1) > (c.debug.logMaxHostSample)) ? (1) : (c.debug.logMaxHostSample));
                    std::vector<float> h_lambda(M);
                    if (M > 0 && m_bufs.d_lambda) {
                        CUDA_CHECK(cudaMemcpy((void*)h_lambda.data(), (const void*)m_bufs.d_lambda, sizeof(float) * M, cudaMemcpyDeviceToHost));
                        lamMin = lamMax = (double)h_lambda[0];
                        double s = 0.0;
                        for (uint32_t i = 0; i < M; ++i) {
                            const double v = (double)h_lambda[i];
                            s += v; lamMin = (((lamMin) < (v)) ? (lamMin) : (v)); lamMax = (((lamMax) > (v)) ? (lamMax) : (v));
                        }
                        lamAvg = (M > 0) ? (s / (double)M) : 0.0;
                    }
                }

                static double s_prevKE = -1.0;
                double KE = 0.0, KE_decay = 0.0;
                if (c.debug.logEnergy) {
                    const uint32_t M = std::min<uint32_t>(m_params.numParticles, (uint32_t)(((1) > (c.debug.logMaxHostSample)) ? (1) : (c.debug.logMaxHostSample)));
                    std::vector<float4> h_vel(M);
                    CUDA_CHECK(cudaMemcpy((void*)h_vel.data(), (const void*)m_bufs.d_vel, sizeof(float4) * M, cudaMemcpyDeviceToHost));
                    double sumv2 = 0.0;
                    for (uint32_t i = 0; i < M; ++i) {
                        const double vx = h_vel[i].x, vy = h_vel[i].y, vz = h_vel[i].z;
                        sumv2 += vx * vx + vy * vy + vz * vz;
                    }
                    const double mp = (double)m_params.particleMass;
                    const double invM = (M > 0) ? (1.0 / (double)M) : 0.0;
                    KE = 0.5 * mp * sumv2 * invM;
                    if (s_prevKE > 0.0) KE_decay = KE /(((1e-18) > (s_prevKE)) ? (1e-18) : (s_prevKE));
                    s_prevKE = KE;
                }

                const double rhoRelPrint = (m_params.restDensity > 0.0f) ? (avgR / (double)m_params.restDensity) : 0.0;
                const double sumWPrint = avgRhoRel;

                std::fprintf(stderr,
                    "[Stab] Frame=%llu | N=%u | dt=%.4g iters=%d | xsph=%.3g rest=%.3g | avgSpeed=%.4g (decay=%.3f) | avgRhoRel=%.4f | sumW=%.4f",
                    (unsigned long long)m_frameIndex, m_params.numParticles,
                    (double)m_params.dt, m_params.solverIters,
                    (double)m_params.xsph_c, (double)m_params.boundaryRestitution,
                    avgV, v_decay, rhoRelPrint, sumWPrint);
                if (c.debug.logCFL) {
                    std::fprintf(stderr, " | CFL(avg)=%.4f", cfl_est_avg);
                }
                if (c.debug.logLambda) {
                    std::fprintf(stderr, " | lambda[min,avg,max]=[%.3g, %.3g, %.3g]", lamMin, lamAvg, lamMax);
                }
                if (c.debug.logEnergy) {
                    std::fprintf(stderr, " | KE=%.4g (decay=%.3f)", KE, KE_decay);
                }
                std::fprintf(stderr, "\n");

                if (m_params.boundaryRestitution > 0.05f) {
                    std::fprintf(stderr, "[Hint] boundaryRestitution=%.3g 偏大，建议接近 0。\n", (double)m_params.boundaryRestitution);
                }
                if (m_params.xsph_c <= 0.0f) {
                    std::fprintf(stderr, "[Hint] 未开启 XSPH，可尝试 xsph_c~[0.02,0.1]。\n");
                }
                if (m_params.solverIters < 4) {
                    std::fprintf(stderr, "[Hint] solverIters=%d 偏低，建议 >=4。\n", m_params.solverIters);
                }
                if (cfl_est_avg > (double)c.sim.cfl * 1.25) {
                    std::fprintf(stderr, "[Hint] CFL(avg)=%.3f 高于目标 %.3f，建议减小 dt 或下调速度。\n",
                        cfl_est_avg, (double)c.sim.cfl);
                }
                if (rhoRelPrint > 1.05) {
                    std::fprintf(stderr, "[Hint] 平均相对密度偏高，可增大迭代或减小 dt。\n");
                }
            }
        }
        ++m_frameIndex;
        // ========== 自适应精度检查（不阻塞核心路径） ========== //
        if (console::Instance().sim.precision.adaptivePrecision) {
            // 构造一个轻量 stats（使用已缓存 m_params 与上一帧速度均值等）
            SimStats s{};
            s.N = m_params.numParticles;
            s.avgRhoRel = (m_params.restDensity > 0.f) ? 1.0:// 占位；真实误差由 computeStats 调用时可覆盖。
            // 若需要更准确的误差，请在调用 computeStats 后再传入（可改为用户层驱动）。
            adaptivePrecisionCheck(s);
        }
        return true;
    }

    void Simulator::kIntegratePred(cudaStream_t s, const SimParams& p) {
        prof::Range r("Phase.Integrate", prof::Color(0x50, 0xA0, 0xFF));
        bool useMP = UseHalfForPosition(p, Stage::Integration, m_bufs) &&
            UseHalfForVelocity(p, Stage::Integration, m_bufs);        if (useMP) {
            ::LaunchIntegratePredMP(m_bufs.d_pos, m_bufs.d_vel, m_bufs.d_pos_pred,
                m_bufs.d_pos_h4, m_bufs.d_vel_h4,
                p.gravity, p.dt, p.numParticles, s);
        }
        else {
            ::LaunchIntegratePred(m_bufs.d_pos, m_bufs.d_vel, m_bufs.d_pos_pred, p.gravity, p.dt, p.numParticles, s);
        }
        ::LaunchBoundary(m_bufs.d_pos_pred, m_bufs.d_vel, p.grid, 0.0f, p.numParticles, s);
    }

    void Simulator::kHashKeys(cudaStream_t s, const SimParams& p) {
        prof::Range r("Phase.HashKeys", prof::Color(0x60, 0xB0, 0x40));
        bool useMP = UseHalfForPosition(p, Stage::GridBuild, m_bufs);
        if (useMP) {
            ::LaunchHashKeysMP(m_bufs.d_cellKeys, m_bufs.d_indices,
                m_bufs.d_pos_pred, m_bufs.d_pos_pred_h4,
                p.grid, p.numParticles, s);
        }
        else {
            ::LaunchHashKeys(m_bufs.d_cellKeys, m_bufs.d_indices, m_bufs.d_pos_pred, p.grid, p.numParticles, s);
        }
    }

    void Simulator::kSort(cudaStream_t s, const SimParams& p) {
        prof::Range r("Phase.SortPairs", prof::Color(0x90, 0x50, 0xF0));
        ::LaunchSortPairs(m_bufs.d_sortTemp, m_bufs.sortTempBytes,
            m_bufs.d_cellKeys, m_bufs.d_cellKeys_sorted,
            m_bufs.d_indices, m_bufs.d_indices_sorted,
            p.numParticles, s);
    }

    void Simulator::kCellRanges(cudaStream_t s, const SimParams& p) {
        prof::Range r("Phase.CellRanges.Dense", prof::Color(0x40, 0xD0, 0xD0));
        CUDA_CHECK(cudaMemsetAsync(m_bufs.d_cellStart, 0xFF, sizeof(uint32_t) * m_numCells, s));
        CUDA_CHECK(cudaMemsetAsync(m_bufs.d_cellEnd, 0xFF, sizeof(uint32_t) * m_numCells, s));
        ::LaunchCellRanges(m_bufs.d_cellStart, m_bufs.d_cellEnd, m_bufs.d_cellKeys_sorted, p.numParticles, m_numCells, s);
    }

    void Simulator::kCellRangesCompact(cudaStream_t s, const SimParams& p) {
        prof::Range r("Phase.CellRanges.Dense", prof::Color(0x40, 0xD0, 0xD0));
        ::LaunchCellRangesCompact(
            m_bufs.d_cellUniqueKeys,
            m_bufs.d_cellOffsets,
            m_bufs.d_compactCount,
            m_bufs.d_cellKeys_sorted,
            p.numParticles,
            s);
        m_numCompactCells = 0;
        (void)p;
    }

    void Simulator::kSolveIter(cudaStream_t s, const SimParams& p) {
        prof::Range rIter("Phase.SolveIter", prof::Color(0xE0, 0x80, 0x40));
        sim::DeviceParams dp = sim::MakeDeviceParams(p);
        if (m_useHashedGrid) {
            bool useMP = UseHalfForPosition(p, Stage::LambdaSolve, m_bufs);
            if (useMP) {
                ::LaunchLambdaCompactMP(
                    m_bufs.d_lambda,
                    m_bufs.d_pos_pred, m_bufs.d_pos_pred_h4,
                    m_bufs.d_indices_sorted, m_bufs.d_cellKeys_sorted,
                    m_bufs.d_cellUniqueKeys, m_bufs.d_cellOffsets, m_bufs.d_compactCount,
                    dp, p.numParticles, s);
                ::LaunchDeltaApplyCompactMP(
                    m_bufs.d_pos_pred, m_bufs.d_delta, m_bufs.d_lambda,
                    m_bufs.d_pos_pred, m_bufs.d_pos_pred_h4,
                    m_bufs.d_indices_sorted, m_bufs.d_cellKeys_sorted,
                    m_bufs.d_cellUniqueKeys, m_bufs.d_cellOffsets, m_bufs.d_compactCount,
                    dp, p.numParticles, s);
            }
            else {
                ::LaunchLambdaCompact(
                    m_bufs.d_lambda, m_bufs.d_pos_pred, m_bufs.d_indices_sorted,
                    m_bufs.d_cellKeys_sorted,
                    m_bufs.d_cellUniqueKeys, m_bufs.d_cellOffsets, m_bufs.d_compactCount,
                    dp, p.numParticles, s);
                ::LaunchDeltaApplyCompact(
                    m_bufs.d_pos_pred, m_bufs.d_delta, m_bufs.d_lambda,
                    m_bufs.d_indices_sorted, m_bufs.d_cellKeys_sorted,
                    m_bufs.d_cellUniqueKeys, m_bufs.d_cellOffsets, m_bufs.d_compactCount,
                    dp, p.numParticles, s);
            }
        } else {
            bool useMP = UseHalfForPosition(p, Stage::LambdaSolve, m_bufs);            
            if (useMP) {
                ::LaunchLambdaMP(
                    m_bufs.d_lambda,
                    m_bufs.d_pos_pred, m_bufs.d_pos_pred_h4,
                    m_bufs.d_indices_sorted,
                    m_bufs.d_cellKeys_sorted,
                    m_bufs.d_cellStart, m_bufs.d_cellEnd,
                    dp, p.numParticles, s);
                ::LaunchDeltaApplyMP(
                    m_bufs.d_pos_pred, m_bufs.d_delta, m_bufs.d_lambda,
                    m_bufs.d_pos_pred, m_bufs.d_pos_pred_h4,
                    m_bufs.d_indices_sorted, m_bufs.d_cellKeys_sorted,
                    m_bufs.d_cellStart, m_bufs.d_cellEnd,
                    dp, p.numParticles, s);
            }
            else {
                ::LaunchLambda(m_bufs.d_lambda, m_bufs.d_pos_pred, m_bufs.d_indices_sorted,
                    m_bufs.d_cellKeys_sorted,
                    m_bufs.d_cellStart, m_bufs.d_cellEnd,
                    dp, p.numParticles, s);
                ::LaunchDeltaApply(m_bufs.d_pos_pred, m_bufs.d_delta, m_bufs.d_lambda,
                    m_bufs.d_indices_sorted,
                    m_bufs.d_cellKeys_sorted,
                    m_bufs.d_cellStart, m_bufs.d_cellEnd,
                    dp, p.numParticles, s);
            }
        }
        ::LaunchBoundary(m_bufs.d_pos_pred, m_bufs.d_vel, p.grid, 0.0f, p.numParticles, s);
    }

    void sim::Simulator::kVelocityAndPost(cudaStream_t s, const SimParams& p) {
        prof::Range rPost("Phase.VelocityPost", prof::Color(0xC0, 0x40, 0xA0));
        bool useMP = UseHalfForPosition(p, Stage::VelocityUpdate, m_bufs);        
        if (useMP) {
            ::LaunchVelocityMP(m_bufs.d_vel, m_bufs.d_pos, m_bufs.d_pos_pred,
                m_bufs.d_pos_h4, m_bufs.d_pos_pred_h4,
                1.0f / p.dt, p.numParticles, s);
        }
        else {
            ::LaunchVelocity(m_bufs.d_vel, m_bufs.d_pos, m_bufs.d_pos_pred, 1.0f / p.dt, p.numParticles, s);
        }
        bool xsphApplied = false;
        if (p.xsph_c > 0.0f && p.numParticles > 0) {
            sim::DeviceParams dp = sim::MakeDeviceParams(p);
            if (m_useHashedGrid) {
                bool useMP = UseHalfForPosition(p, Stage::XSPH, m_bufs) &&
                    UseHalfForVelocity(p, Stage::XSPH, m_bufs);
                if (useMP) {
                    ::LaunchXSPHCompactMP(
                        m_bufs.d_delta,
                        m_bufs.d_vel, m_bufs.d_vel_h4,
                        m_bufs.d_pos_pred, m_bufs.d_pos_pred_h4,
                        m_bufs.d_indices_sorted, m_bufs.d_cellKeys_sorted,
                        m_bufs.d_cellUniqueKeys, m_bufs.d_cellOffsets, m_bufs.d_compactCount,
                        dp, p.numParticles, s);
                }
                else {
                    ::LaunchXSPHCompact(m_bufs.d_delta, m_bufs.d_vel, m_bufs.d_pos_pred,
                        m_bufs.d_indices_sorted, m_bufs.d_cellKeys_sorted,
                        m_bufs.d_cellUniqueKeys, m_bufs.d_cellOffsets, m_bufs.d_compactCount,
                        dp, p.numParticles, s);
                }
            }
            else {
                bool useMP = UseHalfForPosition(p, Stage::XSPH, m_bufs) &&
                    UseHalfForVelocity(p, Stage::XSPH, m_bufs);              
                if (useMP) {
                    ::LaunchXSPHMP(
                        m_bufs.d_delta,
                        m_bufs.d_vel, m_bufs.d_vel_h4,
                        m_bufs.d_pos_pred, m_bufs.d_pos_pred_h4,
                        m_bufs.d_indices_sorted, m_bufs.d_cellKeys_sorted,
                        m_bufs.d_cellStart, m_bufs.d_cellEnd,
                        dp, p.numParticles, s);
                }
                else {
                    ::LaunchXSPH(m_bufs.d_delta, m_bufs.d_vel, m_bufs.d_pos_pred,
                        m_bufs.d_indices_sorted, m_bufs.d_cellKeys_sorted,
                        m_bufs.d_cellStart, m_bufs.d_cellEnd,
                        dp, p.numParticles, s);
                }
            }
            xsphApplied = true;
        }

        float4* effectiveVel = xsphApplied ? m_bufs.d_delta : m_bufs.d_vel;
        ::LaunchBoundary(m_bufs.d_pos_pred, effectiveVel, p.grid, p.boundaryRestitution, p.numParticles, s);
        ::LaunchRecycleToNozzleConst(m_bufs.d_pos, m_bufs.d_pos_pred, effectiveVel, p.grid, p.dt, p.numParticles, 0, s);

        if (p.numParticles > 0) {
            CUDA_CHECK(cudaMemcpyAsync(m_bufs.d_pos, m_bufs.d_pos_pred,
                sizeof(float4) * p.numParticles, cudaMemcpyDeviceToDevice, s));
        }
        if (xsphApplied && p.numParticles > 0) {
            CUDA_CHECK(cudaMemcpyAsync(m_bufs.d_vel, m_bufs.d_delta,
                sizeof(float4) * p.numParticles, cudaMemcpyDeviceToDevice, s));
        }

        // 新增：同步半精镜像（仅存储，不改变现有计算流程）
        if (m_bufs.anyHalf() && m_params.numParticles > 0 && !m_precisionLogged) {
            m_bufs.packAllToHalf(m_params.numParticles, m_stream);
            UpdateDevicePrecisionView(m_bufs, m_params.precision);
            std::fprintf(stderr,
                "[Precision] First emission half-pack done | N=%u | d_pos_h4=%p\n",
                m_params.numParticles, (void*)m_bufs.d_pos_h4);
            m_precisionLogged = true;
        }
    }

    void Simulator::seedBoxLattice(uint32_t nx, uint32_t ny, uint32_t nz, float3 origin, float spacing) {
        const uint64_t nreq64 = uint64_t(nx) * uint64_t(ny) * uint64_t(nz);
        uint32_t Nreq = (nreq64 > UINT32_MAX) ? UINT32_MAX : uint32_t(nreq64);

        uint32_t N = Nreq;
        if (N > m_bufs.capacity) {
            m_bufs.allocate(N);
            std::vector<uint32_t> h_idx(N);
            for (uint32_t i = 0; i < N; ++i) h_idx[i] = i;
            CUDA_CHECK(cudaMemcpy(m_bufs.d_indices, h_idx.data(), sizeof(uint32_t) * N, cudaMemcpyHostToDevice));
            m_graphDirty = true;
        }

        m_params.numParticles = N;

        std::vector<float4> h_pos(N);
        uint32_t idx = 0;
        for (uint32_t iz = 0; iz < nz && idx < N; ++iz) {
            for (uint32_t iy = 0; iy < ny && idx < N; ++iy) {
                for (uint32_t ix = 0; ix < nx && idx < N; ++ix) {
                    float x = origin.x + float(ix) * spacing;
                    float y = origin.y + float(iy) * spacing;
                    float z = origin.z + float(iz) * spacing;
                    x = fminf(fmaxf(x, m_params.grid.mins.x), m_params.grid.maxs.x);
                    y = fminf(fmaxf(y, m_params.grid.mins.y), m_params.grid.maxs.y);
                    z = fminf(fmaxf(z, m_params.grid.mins.z), m_params.grid.maxs.z);
                    h_pos[idx++] = make_float4(x, y, z, 1.0f);
                }
            }
        }
        for (; idx < N; ++idx) h_pos[idx] = (N > 0) ? h_pos[N - 1] : make_float4(origin.x, origin.y, origin.z, 1.0f);

        {
            const auto& c = console::Instance();
            if (c.sim.initial_jitter_enable) {
                const float h_use = (m_params.kernel.h > 0.f) ? m_params.kernel.h : spacing;
                const float amp = c.sim.initial_jitter_scale_h * h_use;
                if (amp > 0.f) {
                    std::mt19937 jrng(c.sim.initial_jitter_seed);
                    std::uniform_real_distribution<float> J(-1.0f, 1.0f);
                    for (uint32_t i = 0; i < N; ++i) {
                        float ox, oy, oz;
                        for (;;) {
                            ox = J(jrng); oy = J(jrng); oz = J(jrng);
                            if (ox * ox + oy * oy + oz * oz <= 1.0f) break;
                        }
                        ox *= amp; oy *= amp; oz *= amp;
                        float4 p = h_pos[i];
                        p.x += ox; p.y += oy; p.z += oz;
                        p.x = fminf(fmaxf(p.x, m_params.grid.mins.x), m_params.grid.maxs.x);
                        p.y = fminf(fmaxf(p.y, m_params.grid.mins.y), m_params.grid.maxs.y);
                        p.z = fminf(fmaxf(p.z, m_params.grid.mins.z), m_params.grid.maxs.z);
                        h_pos[i] = p;
                    }
                }
            }
        }

        CUDA_CHECK(cudaMemcpy(m_bufs.d_pos, h_pos.data(), sizeof(float4) * N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m_bufs.d_pos_pred, h_pos.data(), sizeof(float4) * N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(m_bufs.d_vel, 0, sizeof(float4) * N));
        if (m_bufs.anyHalf()) {
            m_bufs.packAllToHalf(N, 0);
        }
    }

    void Simulator::seedBoxLatticeAuto(uint32_t total, float3 origin, float spacing) {
        float3 size = make_float3(
            fmaxf(0.f, m_params.grid.maxs.x - m_params.grid.mins.x),
            fmaxf(0.f, m_params.grid.maxs.y - m_params.grid.mins.y),
            fmaxf(0.f, m_params.grid.maxs.z - m_params.grid.mins.z)
        );
        auto cap = [&spacing](float len) -> uint32_t {
            return (spacing > 0.f) ? (uint32_t)(((0) > ((int)floorf(len / spacing))) ? (0) : ((int)floorf(len / spacing))) : 0u;
            };
        const uint32_t maxX = (((1u) > (cap(size.x))) ? (1u) : (cap(size.x)));
        const uint32_t maxY = (((1u) > (cap(size.y))) ? (1u) : (cap(size.y)));
        const uint32_t maxZ = (((1u) > (cap(size.z))) ? (1u) : (cap(size.z)));
        uint64_t T = std::max<uint32_t>(1u, total);
        uint32_t nx = std::min<uint32_t>(maxX, (uint32_t)ceil(pow((double)T, 1.0 / 3.0)));
        uint64_t remXY = (T + nx - 1) / nx;
        uint32_t ny = std::min<uint32_t>(maxY, (uint32_t)ceil(sqrt((double)remXY)));
        uint32_t nz = std::min<uint32_t>(maxZ, (uint32_t)ceil((double)remXY / std::max<uint32_t>(1u, ny)));

        auto safeMul = [](uint64_t a, uint64_t b) { return a > 0 && b > (UINT64_MAX / a) ? UINT64_MAX : a * b; };
        while (safeMul(nx, safeMul(ny, nz)) < T) {
            if (nx < maxX) ++nx;
            else if (ny < maxY) ++ny;
            else if (nz < maxZ) ++nz;
            else break;
        }

        uint64_t nreq64 = (uint64_t)nx * (uint64_t)ny * (uint64_t)nz;
        const uint32_t Nreq = (nreq64 > UINT32_MAX) ? UINT32_MAX : (uint32_t)nreq64;
        if (Nreq < total) {
            const auto& c = console::Instance();
            if (c.debug.printWarnings) {
                std::fprintf(stderr, "[Warn] seedBoxLatticeAuto: Nreq(%u) < total(%u). Potential overlap.\n", Nreq, total);
            }
        }
        float3 margin = make_float3(spacing * 0.25f, spacing * 0.25f, spacing * 0.25f);
        origin.x = fminf(fmaxf(origin.x, m_params.grid.mins.x + margin.x), m_params.grid.maxs.x - margin.x);
        origin.y = fminf(fmaxf(origin.y, m_params.grid.mins.y + margin.y), m_params.grid.maxs.y - margin.y);
        origin.z = fminf(fmaxf(origin.z, m_params.grid.mins.z + margin.z), m_params.grid.maxs.z - margin.z);

        seedBoxLattice(nx, ny, nz, origin, spacing);
    }

    bool Simulator::importPosPredFromD3D12(void* sharedHandleWin32, size_t bytes) {
        if (!sharedHandleWin32 || bytes == 0) return false;
        if (m_extPosPred) {
            cudaDestroyExternalMemory(m_extPosPred);
            m_extPosPred = nullptr;
            m_bufs.detachExternalPosPred();
        }
        cudaExternalMemoryHandleDesc memDesc{};
        memDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
        memDesc.handle.win32.handle = sharedHandleWin32;
        memDesc.size = bytes;
        memDesc.flags = cudaExternalMemoryDedicated;
        CUDA_CHECK(cudaImportExternalMemory(&m_extPosPred, &memDesc));
        cudaExternalMemoryBufferDesc bufDesc{};
        bufDesc.offset = 0;
        bufDesc.size = bytes;
        bufDesc.flags = 0;
        void* devPtr = nullptr;
        CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&devPtr, m_extPosPred, &bufDesc));
        m_bufs.bindExternalPosPred(reinterpret_cast<float4*>(devPtr));
        m_canPingPongPos = true;
        return true;
    }

    bool Simulator::computeStats(SimStats& out, uint32_t sampleStride) const {
        out = {};
        if (m_params.numParticles == 0 || m_bufs.capacity == 0) { return true; }
        const auto& c = console::Instance();
        if (m_useHashedGrid) {
            LaunchCellRanges(
                m_bufs.d_cellStart,
                m_bufs.d_cellEnd,
                m_bufs.d_cellKeys_sorted,
                m_params.numParticles,
                m_numCells,
                m_stream);
            CUDA_CHECK(cudaStreamSynchronize(m_stream));
        }
        double avgN_grid = 0.0, avgV = 0.0, avgRhoRel_grid = 0.0, avgR_grid = 0.0;
        const uint32_t stride = (sampleStride == 0 ? 1u : sampleStride);
        bool ok_grid = LaunchComputeStats(
            m_bufs.d_pos_pred,
            m_bufs.d_vel,
            m_bufs.d_indices_sorted,
            m_bufs.d_cellStart,
            m_bufs.d_cellEnd,
            m_params.grid,
            m_params.kernel,
            m_params.particleMass,
            m_params.numParticles,
            m_numCells,
            stride,
            &avgN_grid, &avgV, &avgRhoRel_grid, &avgR_grid,
            m_stream);
        if (!ok_grid) return false;
        out.N = m_params.numParticles;
        out.avgNeighbors = avgN_grid;
        out.avgSpeed = avgV;
        const double rhoRel_grid = (m_params.restDensity > 0.0f) ? (avgR_grid / (double)m_params.restDensity) : 0.0;
        out.avgRhoRel = rhoRel_grid;
        out.avgRho = avgR_grid;

        const bool wantDiag = (c.debug.printDiagnostics || c.debug.printWarnings || c.debug.printHints);
        const uint64_t diagEvery = (uint64_t)(c.debug.logEveryN <= 0 ? 1 : c.debug.logEveryN);
        if (!wantDiag || ((m_frameIndex % diagEvery) != 0ull)) return true;

        double avgN_bf = 0.0, avgV_bf = 0.0, avgRhoRel_bf = 0.0, avgR_bf = 0.0;
        const uint32_t kMaxISamples = 2048;
        bool ok_bf = LaunchComputeStatsBruteforce(
            m_bufs.d_pos_pred,
            m_bufs.d_vel,
            m_params.kernel,
            m_params.particleMass,
            m_params.numParticles,
            stride,
            kMaxISamples,
            &avgN_bf, &avgV_bf, &avgRhoRel_bf, &avgR_bf,
            m_stream);
        if (!ok_bf) return true;

        const bool hasNeighborCap = (m_params.maxNeighbors > 0);
        const double capN = hasNeighborCap ? double(m_params.maxNeighbors) : std::numeric_limits<double>::infinity();
        const double nearCapRatio = 0.9;
        const bool nearCap = hasNeighborCap && (avgN_grid >= nearCapRatio * capN);
        const double nAbsThresh = 2.0;
        const double nRelThresh = 0.5;
        const bool nSeverelyUndercount =
            (avgN_bf > 0.0) && ((avgN_grid + nAbsThresh) < avgN_bf * (1.0 - nRelThresh));
        const double rhoRel_bf = (m_params.restDensity > 0.0f) ? (avgR_bf / (double)m_params.restDensity) : 0.0;

        static uint64_t s_lastDiagFrame = UINT64_MAX;
        const bool shouldDiag = (nSeverelyUndercount || nearCap);
        if (shouldDiag && s_lastDiagFrame != m_frameIndex &&
            (c.debug.printDiagnostics || c.debug.printWarnings || c.debug.printHints)) {
            s_lastDiagFrame = m_frameIndex;
            const double h = double(m_params.kernel.h);
            const double cell = double(m_params.grid.cellSize);
            const double ratio_cell_h = (h > 0.0) ? (cell / h) : 0.0;
            if (c.debug.printDiagnostics) {
                std::fprintf(stderr,
                    "[Diag] Frame=%llu | N=%u | h=%.6g | cell=%.6g (cell/h=%.3f) | dim=(%d,%d,%d) numCells=%u\n",
                    (unsigned long long)m_frameIndex, m_params.numParticles, h, cell, ratio_cell_h,
                    m_params.grid.dim.x, m_params.grid.dim.y, m_params.grid.dim.z, m_numCells);
                std::fprintf(stderr,
                    "[Diag] Neighbors: grid=%.3f, brute=%.3f | RhoRel: grid=%.3f, brute=%.3f | maxNeighbors=%d%s\n",
                    avgN_grid, avgN_bf, rhoRel_grid, rhoRel_bf, m_params.maxNeighbors,
                    (nearCap ? " (near cap)" : ""));
            }
            if (c.debug.printHints) {
                if (ratio_cell_h < 0.9) std::fprintf(stderr, "[Hint] cellSize/h <0.9，建议设 ~[1.0,1.5]h。\n");
                else if (ratio_cell_h > 2.0) std::fprintf(stderr, "[Hint] cellSize/h >2.0，建议设 ~[1.0,1.5]h。\n");
                if (nearCap) std::fprintf(stderr, "[Hint] avgNeighbors 接近上限 %d，考虑调大或置 0。\n", m_params.maxNeighbors);
                if (nSeverelyUndercount) std::fprintf(stderr, "[Hint] 网格邻居显著偏低，检查 cellSize/h 与邻域遍历。\n");
            }
            if (c.debug.printDiagnostics) {
                CUDA_CHECK(cudaStreamSynchronize(m_stream));
                std::vector<uint32_t> h_start(m_numCells), h_end(m_numCells);
                CUDA_CHECK(cudaMemcpy(h_start.data(), m_bufs.d_cellStart, sizeof(uint32_t) * m_numCells, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_end.data(), m_bufs.d_cellEnd, sizeof(uint32_t) * m_numCells, cudaMemcpyDeviceToHost));
                const uint32_t EMPTY = 0xFFFFFFFFu;
                uint64_t nonEmpty = 0, empty = 0;
                uint64_t sumCnt = 0, sumCnt2 = 0;
                uint32_t minCnt = UINT32_MAX, maxCnt = 0, maxIdx = 0;
                for (uint32_t i = 0; i < m_numCells; ++i) {
                    const uint32_t s = h_start[i], e = h_end[i];
                    if (s == EMPTY || e == EMPTY || e < s) { ++empty; continue; }
                    const uint32_t cnt = (e - s);
                    ++nonEmpty;
                    sumCnt += cnt;
                    sumCnt2 += uint64_t(cnt) * uint64_t(cnt);
                    if (cnt < minCnt) minCnt = cnt;
                    if (cnt > maxCnt) { maxCnt = cnt; maxIdx = i; }
                }
                const double meanCnt = (nonEmpty > 0) ? double(sumCnt) / double(nonEmpty) : 0.0;
                const double varCnt = (nonEmpty > 0) ? (double(sumCnt2) / double(nonEmpty) - meanCnt * meanCnt) : 0.0;
                const double stdCnt = (varCnt > 0.0) ? std::sqrt(varCnt) : 0.0;
                auto idxTo3D = [dim = m_params.grid.dim](uint32_t lid) {
                    int3 c;
                    c.x = int(lid % uint32_t(dim.x));
                    const uint32_t xy = uint32_t(dim.x) * uint32_t(dim.y);
                    c.y = int((lid / uint32_t(dim.x)) % uint32_t(dim.y));
                    c.z = int(lid / xy);
                    return c;
                    };
                const int3 cMax = idxTo3D(maxIdx);
                const double fracMaxCell = (m_params.numParticles > 0) ? (double(maxCnt) / double(m_params.numParticles)) : 0.0;
                std::fprintf(stderr,
                    "[Diag] Cell occupancy: empty=%llu (%.2f%%), non-empty=%llu | cnt[min= %u, avg=%.2f, max=%u (cell=(%d,%d,%d), %.2f%%)] | std=%.2f\n",
                    (unsigned long long)empty, 100.0 * double(empty) / double(m_numCells),
                    (unsigned long long)nonEmpty,
                    (unsigned)(nonEmpty ? minCnt : 0u), meanCnt, (unsigned)maxCnt,
                    cMax.x, cMax.y, cMax.z, 100.0 * fracMaxCell, stdCnt);
            }
        }
        return true;
    }

    bool Simulator::computeStatsBruteforce(SimStats& out, uint32_t sampleStride, uint32_t maxISamples) const {
        if (m_params.numParticles == 0) { out = {}; return true; }
        double avgN = 0.0, avgV = 0.0, avgRhoRel = 0.0, avgR = 0.0;
        bool ok = LaunchComputeStatsBruteforce(
            m_bufs.d_pos_pred,
            m_bufs.d_vel,
            m_params.kernel,
            m_params.particleMass,
            m_params.numParticles,
            (sampleStride == 0 ? 1u : sampleStride),
            maxISamples,
            &avgN, &avgV, &avgRhoRel, &avgR,
            m_stream);
        if (!ok) return false;
        out.N = m_params.numParticles;
        out.avgNeighbors = avgN;
        out.avgSpeed = avgV;
        const double rhoRel_bf = (m_params.restDensity > 0.0f) ? (avgR / (double)m_params.restDensity) : 0.0;
        out.avgRhoRel = rhoRel_bf;
        out.avgRho = avgR;
        return true;
    }

    bool Simulator::computeCenterOfMass(float3& outCom, uint32_t sampleStride) const {
        outCom = make_float3(0, 0, 0);
        const uint32_t N = m_params.numParticles;
        if (N == 0) return true;
        std::vector<float4> h_pos(N);
        CUDA_CHECK(cudaMemcpy((void*)h_pos.data(), (const void*)m_bufs.d_pos_pred, sizeof(float4) * N, cudaMemcpyDeviceToHost));
        uint64_t cnt = 0;
        double sx = 0.0, sy = 0.0, sz = 0.0;
        const uint32_t stride = (sampleStride == 0 ? 1u : sampleStride);
        for (uint32_t i = 0; i < N; i += stride) {
            sx += h_pos[i].x; sy += h_pos[i].y; sz += h_pos[i].z;
            ++cnt;
        }
        if (cnt == 0) return true;
        const double inv = 1.0 / double(cnt);
        outCom = make_float3(float(sx * inv), float(sy * inv), float(sz * inv));
        return true;
    }

    void Simulator::seedCubeMix(uint32_t groupCount,
        const float3* groupCenters,
        uint32_t edgeParticles,
        float spacing,
        bool applyJitter,
        float jitterAmp,
        uint32_t jitterSeed)
    {
        if (groupCount == 0 || edgeParticles == 0) {
            m_params.numParticles = 0;
            return;
        }
        uint64_t per = (uint64_t)edgeParticles * edgeParticles * edgeParticles;
        uint64_t total64 = per * (uint64_t)groupCount;
        uint32_t total = (total64 > UINT32_MAX) ? UINT32_MAX : (uint32_t)total64;

        if (total > m_bufs.capacity) {
            m_bufs.allocate(total);
            std::vector<uint32_t> h_idx(total);
            for (uint32_t i = 0; i < total; ++i) h_idx[i] = i;
            CUDA_CHECK(cudaMemcpy(m_bufs.d_indices, h_idx.data(), sizeof(uint32_t) * total, cudaMemcpyHostToDevice));
            m_graphDirty = true;
        }
        m_params.numParticles = total;

        std::vector<float4> h_pos(total);
        std::vector<float4> h_vel(total, make_float4(0, 0, 0, 0));

        uint32_t cursor = 0;
        for (uint32_t g = 0; g < groupCount && cursor < total; ++g) {
            float3 c = groupCenters[g];
            // 左下角起点（使粒子团包围盒中心落在 c）
            float startOffset = -0.5f * spacing * float(edgeParticles - 1);
            for (uint32_t iz = 0; iz < edgeParticles && cursor < total; ++iz) {
                for (uint32_t iy = 0; iy < edgeParticles && cursor < total; ++iy) {
                    for (uint32_t ix = 0; ix < edgeParticles && cursor < total; ++ix) {
                        float3 p = make_float3(
                            c.x + startOffset + ix * spacing,
                            c.y + startOffset + iy * spacing,
                            c.z + startOffset + iz * spacing);
                        p.x = fminf(fmaxf(p.x, m_params.grid.mins.x), m_params.grid.maxs.x);
                        p.y = fminf(fmaxf(p.y, m_params.grid.mins.y), m_params.grid.maxs.y);
                        p.z = fminf(fmaxf(p.z, m_params.grid.mins.z), m_params.grid.maxs.z);
                        // pos.w 写入“团索引”供着色器使用
                        h_pos[cursor++] = make_float4(p.x, p.y, p.z, (float)g);
                    }
                }
            }
        }
        // 抖动
        if (applyJitter && jitterAmp > 0.f) {
            std::mt19937 jrng(jitterSeed);
            std::uniform_real_distribution<float> U(-1.f, 1.f);
            for (uint32_t i = 0; i < total; ++i) {
                float ox, oy, oz;
                for (;;) {
                    ox = U(jrng); oy = U(jrng); oz = U(jrng);
                    if (ox * ox + oy * oy + oz * oz <= 1.0f) break;
                }
                ox *= jitterAmp; oy *= jitterAmp; oz *= jitterAmp;
                float4 p = h_pos[i];
                p.x = fminf(fmaxf(p.x + ox, m_params.grid.mins.x), m_params.grid.maxs.x);
                p.y = fminf(fmaxf(p.y + oy, m_params.grid.mins.y), m_params.grid.maxs.y);
                p.z = fminf(fmaxf(p.z + oz, m_params.grid.mins.z), m_params.grid.maxs.z);
                h_pos[i] = p;
            }
        }

        CUDA_CHECK(cudaMemcpy(m_bufs.d_pos, h_pos.data(), sizeof(float4) * total, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m_bufs.d_pos_pred, h_pos.data(), sizeof(float4) * total, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m_bufs.d_vel, h_vel.data(), sizeof(float4) * total, cudaMemcpyHostToDevice));
        if (m_bufs.anyHalf()) {
            m_bufs.packAllToHalf(total, 0);
        }
    }
} // namespace sim