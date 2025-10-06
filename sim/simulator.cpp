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
// 新增：集中从控制台获取发射/回收参数
#include "../engine/core/console.h"

// —— 内核包装的全局声明（与 .cu 文件保持一致的 extern "C" + 全局命名空间） 
extern "C" void LaunchIntegratePred(float4* pos, const float4* vel, float4* pos_pred, float3 gravity, float dt, uint32_t N, cudaStream_t s); 
extern "C" void LaunchHashKeys(uint32_t* keys, uint32_t* indices, const float4* pos, sim::GridBounds grid, uint32_t N, cudaStream_t s); 
extern "C" void LaunchCellRanges(uint32_t* cellStart, uint32_t* cellEnd, const uint32_t* keysSorted, uint32_t N, uint32_t numCells, cudaStream_t s); 
extern "C" void LaunchLambda(
    float* lambda,
    const float4* pos_pred,
    const uint32_t* indicesSorted,
    const uint32_t* cellStart,
    const uint32_t* cellEnd,
    sim::DeviceParams dp,
    uint32_t N,
    cudaStream_t s);

extern "C" void LaunchDeltaApply(
    float4* pos_pred,
    float4* delta,
    const float* lambda,
    const uint32_t* indicesSorted,
    const uint32_t* cellStart,
    const uint32_t* cellEnd,
    sim::DeviceParams dp,
    uint32_t N,
    cudaStream_t s); 
extern "C" void LaunchVelocity(float4* vel, const float4* pos, const float4* pos_pred, float inv_dt, uint32_t N, cudaStream_t s);
// 保持与工程中 boundary.cu 的签名一致（包含 restitution） 
extern "C" void LaunchBoundary(float4* pos_pred, float4* vel, sim::GridBounds grid, float restitution, uint32_t N, cudaStream_t s);
// —— CUB radix sort（与 sort_pairs.cu 完全一致的 C 接口） —— 
extern "C" void LaunchSortPairsQuery(size_t* tempBytes, const uint32_t* d_keys_in, uint32_t* d_keys_out, const uint32_t* d_vals_in, uint32_t* d_vals_out, uint32_t N, cudaStream_t s); 
extern "C" void LaunchSortPairs(void* d_temp_storage, size_t tempBytes, uint32_t* d_keys_in, uint32_t* d_keys_out, uint32_t* d_vals_in, uint32_t* d_vals_out, uint32_t N, cudaStream_t s);
// 回收到喷口（常量参数 wrapper + 获取设备内核函数指针） 
extern "C" void LaunchRecycleToNozzleConst(float4* pos, float4* pos_pred, float4* vel, sim::GridBounds grid, float dt, uint32_t N, int enabled, cudaStream_t s); 
extern "C" void* GetRecycleKernelPtr();
// 新增：统计包装
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
// 新增：暴力版统计接口声明
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
extern "C" void LaunchXSPH(
    float4* vel_out,
    const float4* vel_in,
    const float4* pos_pred,
    const uint32_t* indicesSorted,
    const uint32_t* cellStart,
    const uint32_t* cellEnd,
    sim::DeviceParams dp,
    uint32_t N,
    cudaStream_t s);

// 新增：Poisson-disk 采样（2D 圆盘，Bridson 算法）
namespace {
    static inline float2 f2_add(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
    static inline float  f2_len2(float2 a) { return a.x * a.x + a.y * a.y; }

    // 在半径区间 [rMin, 2*rMin] 的环形区均匀采样（面积均匀）
    static inline float2 sample_annulus(float rMin, std::mt19937& rng) {
        std::uniform_real_distribution<float> U(0.0f, 1.0f);
        const float R0 = rMin;
        const float R1 = 2.0f * rMin;
        const float u = U(rng);
        const float r = std::sqrt(R0 * R0 + u * (R1 * R1 - R0 * R0));
        const float th = 6.28318530718f * U(rng); // 2*pi
        return make_float2(r * std::cos(th), r * std::sin(th));
    }

    // 在半径 R 的圆盘内均匀随机采样一个点
    static inline float2 sample_in_disk(float R, std::mt19937& rng) {
        std::uniform_real_distribution<float> U(0.0f, 1.0f);
        const float r = R * std::sqrt(U(rng));
        const float th = 6.28318530718f * U(rng);
        return make_float2(r * std::cos(th), r * std::sin(th));
    }

    // Bridson Poisson-disk for circle domain
    static void poisson_disk_in_circle(float R, float rMin, int target,
        std::mt19937& rng,
        std::vector<float2>& outPts) {
        outPts.clear();
        if (R <= 0.0f || rMin <= 0.0f || target <= 0) return;

        const float cell = rMin / 1.41421356237f; // rMin / sqrt(2)
        const float side = 2.0f * R;
        const int gw = (((1) > ((int)std::ceil(side / cell))) ? (1) : ((int)std::ceil(side / cell)));
        const int gh = (((1) > ((int)std::ceil(side / cell))) ? (1) : ((int)std::ceil(side / cell)));
        const int nCells = gw * gh;

        std::vector<int> grid(nCells, -1);
        std::vector<int> active;
        active.reserve(target);
        outPts.reserve(target);

        auto toGrid = [&](const float2& p, int& gx, int& gy) {
            const float x = p.x + R; // shift to [0, 2R]
            const float y = p.y + R;
            gx = std::clamp((int)std::floor(x / cell), 0, gw - 1);
            gy = std::clamp((int)std::floor(y / cell), 0, gh - 1);
            };
        auto gridIdx = [&](int gx, int gy) { return gy * gw + gx; };

        // 放一个初始点
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
                if (f2_len2(q) > R * R) continue; // 不在圆盘内

                int gx, gy;
                toGrid(q, gx, gy);

                // 邻域检查（5x5 覆盖 > rMin）
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

                // 接受
                const int newIndex = (int)outPts.size();
                outPts.push_back(q);
                active.push_back(newIndex);
                grid[gridIdx(gx, gy)] = newIndex;
                found = true;
            }

            if (!found) {
                // 该活动点枯竭
                active[aidx] = active.back();
                active.pop_back();
            }
        }
    }
} // anonymous namespace

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
    static inline bool kernelEqual(const KernelCoeffs& a, const KernelCoeffs& b, float eps = 1e-6f) {
        return approxEq(a.h, b.h, eps) && approxEq(a.inv_h, b.inv_h, eps) && approxEq(a.h2, b.h2, eps)
            && approxEq(a.poly6, b.poly6, eps) && approxEq(a.spiky, b.spiky, eps) && approxEq(a.visc, b.visc, eps);
    }

    // 删除：本地发射参数拼装，改由 console::BuildEmitParams 统一生成
    // inline void FillEmitParamsFromSim(...){...}

    bool Simulator::initialize(const SimParams& p) {
        m_params = p;
        CUDA_CHECK(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&m_evStart));
        CUDA_CHECK(cudaEventCreate(&m_evEnd));

        // 按容量预分配（maxParticles=0 则使用初始 N）
        const uint32_t capacity = (p.maxParticles > 0) ? p.maxParticles : p.numParticles;
        m_bufs.allocate(capacity);

        // 初始化网格（CellStart/End 尺寸）并回填 dim
        if (!buildGrid(m_params)) return false;

        // 初始化 indices = [0..capacity)
        std::vector<uint32_t> h_idx(capacity);
        for (uint32_t i = 0; i < capacity; ++i) h_idx[i] = i;
        CUDA_CHECK(cudaMemcpy(m_bufs.d_indices, h_idx.data(), sizeof(uint32_t) * capacity, cudaMemcpyHostToDevice));

        // 确保 pos_pred 初始与 pos 同步（仅拷贝当前活动 N）
        if (p.numParticles > 0) {
            CUDA_CHECK(cudaMemcpy(m_bufs.d_pos_pred, m_bufs.d_pos, sizeof(float4) * p.numParticles, cudaMemcpyDeviceToDevice));
        }

        m_graphDirty = true;
        m_captured = {}; // 清空快照
        return true;
    }

    void Simulator::shutdown() {
        if (m_extPosPred) {
            cudaDestroyExternalMemory(m_extPosPred);
            m_extPosPred = nullptr;
            m_bufs.detachExternalPosPred();
        }
        if (m_graphExec) { cudaGraphExecDestroy(m_graphExec); m_graphExec = nullptr; }
        if (m_graph) { cudaGraphDestroy(m_graph); m_graph = nullptr; }
        if (m_evStart) { cudaEventDestroy(m_evStart); m_evStart = nullptr; }
        if (m_evEnd) { cudaEventDestroy(m_evEnd);   m_evEnd = nullptr; }
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

        // 回填到运行时参数，供后续内核使用
        m_params.grid.mins = p.grid.mins;
        m_params.grid.maxs = p.grid.maxs;
        m_params.grid.cellSize = p.grid.cellSize;
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

        // 始终回填最新 dim（即使未变化），确保传给内核的 p.grid.dim 正确
        m_params.grid.mins = p.grid.mins;
        m_params.grid.maxs = p.grid.maxs;
        m_params.grid.cellSize = p.grid.cellSize;
        m_params.grid.dim = dim;

        return gridGeomChanged;
    }

    bool Simulator::ensureSortTemp(std::size_t bytes) {
        m_bufs.ensureSortTemp(bytes);
        return true;
    }

    bool Simulator::needsGraphRebuild(const SimParams& p) const {
        if (!m_graphExec) return true;

        if (p.solverIters != m_captured.solverIters) return true;
        if (p.numParticles != m_captured.numParticles) return true;
        if (p.maxNeighbors != m_captured.maxNeighbors) return true;
        if (p.sortEveryN != m_captured.sortEveryN) return true;

        int3 dim;
        dim.x = int(ceilf((p.grid.maxs.x - p.grid.mins.x) / p.grid.cellSize));
        dim.y = int(ceilf((p.grid.maxs.y - p.grid.mins.y) / p.grid.cellSize));
        dim.z = int(ceilf((p.grid.maxs.z - p.grid.mins.z) / p.grid.cellSize));
        uint32_t newNumCells = uint32_t(dim.x) * uint32_t(dim.y) * uint32_t(dim.z);
        if (newNumCells != m_captured.numCells) return true;
        if (!gridEqual(p.grid, m_captured.grid)) return true;

        if (!kernelEqual(p.kernel, m_captured.kernel)) return true;
        if (!approxEq(p.dt, m_captured.dt)) return true;
        if (!approxEq3(p.gravity, m_captured.gravity)) return true;
        if (!approxEq(p.restDensity, m_captured.restDensity)) return true;

        return false;
    }

    bool Simulator::captureGraphIfNeeded(const SimParams& p) {
        if (!m_graphDirty) return true;
        if (m_graphExec) { cudaGraphExecDestroy(m_graphExec); m_graphExec = nullptr; }
        if (m_graph) { cudaGraphDestroy(m_graph); m_graph = nullptr; }

        {
            size_t tempBytes = 0;
            ::LaunchSortPairsQuery(&tempBytes,
                m_bufs.d_cellKeys, m_bufs.d_cellKeys_sorted,
                m_bufs.d_indices, m_bufs.d_indices_sorted,
                p.numParticles, m_stream);
            ensureSortTemp(tempBytes);
        }

        CUDA_CHECK(cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeGlobal));

        kIntegratePred(m_stream, p);
        kHashKeys(m_stream, p);
        kSort(m_stream, p);
        kCellRanges(m_stream, p);

        for (int i = 0; i < p.solverIters; ++i) {
            kSolveIter(m_stream, p);
        }
        kVelocityAndPost(m_stream, p); // 内含回收节点的 wrapper 调用

        CUDA_CHECK(cudaStreamEndCapture(m_stream, &m_graph));
        CUDA_CHECK(cudaGraphInstantiate(&m_graphExec, m_graph, nullptr, nullptr, 0));
        m_graphDirty = false;

        // 枚举节点并匹配回收内核
        m_nodeRecycle = nullptr;
        {
            void* target = GetRecycleKernelPtr(); // 设备内核指针
            size_t numNodes = 0;
            CUDA_CHECK(cudaGraphGetNodes(m_graph, nullptr, &numNodes));
            if (numNodes > 0) {
                std::vector<cudaGraphNode_t> nodes(numNodes);
                CUDA_CHECK(cudaGraphGetNodes(m_graph, nodes.data(), &numNodes));
                for (auto n : nodes) {
                    cudaGraphNodeType t;
                    CUDA_CHECK(cudaGraphNodeGetType(n, &t));
                    if (t != cudaGraphNodeTypeKernel) continue;
                    cudaKernelNodeParams kp{};
                    CUDA_CHECK(cudaGraphKernelNodeGetParams(n, &kp));
                    if (kp.func == target) {
                        m_nodeRecycle = n;
                        break;
                    }
                }
            }
        }

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
        // 同步最新参数到 m_params，并确保 grid.dim 回填
        m_params = p;
        {
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

        // 注满模式：每帧批量发射（在容量范围内），并初始化新粒子的 pos/pos_pred/vel
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

                    std::vector<float4> h_pos(emit);
                    std::vector<float4> h_vel(emit);

                    // —— 圆盘内 Poisson-disk 采样 —— 
                    const float3 dir = normalize3(ep.nozzleDir);
                    // 构造与 dir 正交的基 u,v
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

                    // 若 Poisson 无法凑满 emit（半径太小/间距太大等），余量回退到均匀随机
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
                        // 面积均匀：回退为圆盘内均匀随机，保证发射速率
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

                    // 写入设备（pos 与 pos_pred 初始一致）
                    CUDA_CHECK(cudaMemcpyAsync(m_bufs.d_pos + begin, h_pos.data(), sizeof(float4) * emit, cudaMemcpyHostToDevice, m_stream));
                    CUDA_CHECK(cudaMemcpyAsync(m_bufs.d_pos_pred + begin, h_pos.data(), sizeof(float4) * emit, cudaMemcpyHostToDevice, m_stream));
                    CUDA_CHECK(cudaMemcpyAsync(m_bufs.d_vel + begin, h_vel.data(), sizeof(float4) * emit, cudaMemcpyHostToDevice, m_stream));

                    // 更新活动粒子数
                    m_params.numParticles = end;
                }
            }
        }

        // 若仍不足容量则扩容（兜底）
        if (m_params.numParticles > m_bufs.capacity) {
            m_bufs.allocate(m_params.numParticles);
            std::vector<uint32_t> h_idx(m_params.numParticles);
            for (uint32_t i = 0; i < m_params.numParticles; ++i) h_idx[i] = i;
            CUDA_CHECK(cudaMemcpy(m_bufs.d_indices, h_idx.data(), sizeof(uint32_t) * m_params.numParticles, cudaMemcpyHostToDevice));
            m_graphDirty = true;
        }

        // 每帧：更新设备常量（喷口/回收参数）
        {
            SetEmitParamsAsync(&ep, m_stream);
        }

        updateGridIfNeeded(m_params);
        if (needsGraphRebuild(m_params)) {
            m_graphDirty = true;
        }
        if (!captureGraphIfNeeded(m_params) || !m_graphExec) {
            fprintf(stderr, "Simulator::step: CUDA Graph not ready (capture failed or m_graphExec == nullptr)\n");
            return false;
        }

        // 若捕获到了回收节点：用 SetParams 动态更新其标量参数（enabled/N/dt/喷口参数）
        if (m_nodeRecycle) {
            struct RecycleArgs {
                float4* pos;
                float4* pos_pred;
                float4* vel;
                sim::GridBounds grid;
                float3 nozzlePos;
                float3 nozzleDir;
                float  nozzleRadius;
                float  initSpeed;
                float  dt;
                float  recycleY;
                unsigned int N;
                int enabled;
            };
            static RecycleArgs args; // 静态存放，生命周期覆盖整个运行期

            // 注意顺序必须与设备端核函数参数顺序完全一致
            static void* params[12] = {
                &args.pos, &args.pos_pred, &args.vel,
                &args.grid, &args.nozzlePos, &args.nozzleDir,
                &args.nozzleRadius, &args.initSpeed, &args.dt,
                &args.recycleY, &args.N, &args.enabled
            };

            // 更新参数值
            args.pos = m_bufs.d_pos;
            args.pos_pred = m_bufs.d_pos_pred;
            args.vel = m_bufs.d_vel;
            args.grid = m_params.grid;

            // 统一使用控制台构建的 ep
            args.nozzlePos = ep.nozzlePos;
            args.nozzleDir = ep.nozzleDir;
            args.nozzleRadius = (ep.nozzleRadius > 0.f) ? ep.nozzleRadius : (m_params.kernel.h * 1.5f);
            args.initSpeed = (ep.nozzleSpeed > 0.f) ? ep.nozzleSpeed : 1.0f;
            args.dt = m_params.dt;
            args.recycleY = ep.recycleY;

            args.N = m_params.numParticles;
            args.enabled = (recycleToNozzle ? 1 : 0);

            // 合法的 grid 配置（N==0 时 gridDim 不能为 0）
            uint32_t blocks = (args.N + 255u) / 256u;
            if (blocks == 0u) {
                blocks = 1u;
                args.enabled = 0;
            }

            cudaKernelNodeParams kp{};
            kp.func = GetRecycleKernelPtr();
            kp.gridDim = dim3(blocks, 1, 1);
            kp.blockDim = dim3(256, 1, 1);
            kp.sharedMemBytes = 0;
            kp.kernelParams = params;
            kp.extra = nullptr;

            CUDA_CHECK(cudaGraphExecKernelNodeSetParams(m_graphExec, m_nodeRecycle, &kp));
        }

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
        ::LaunchIntegratePred(m_bufs.d_pos, m_bufs.d_vel, m_bufs.d_pos_pred, p.gravity, p.dt, p.numParticles, s);
    }

    void Simulator::kHashKeys(cudaStream_t s, const SimParams& p) {
        ::LaunchHashKeys(m_bufs.d_cellKeys, m_bufs.d_indices, m_bufs.d_pos_pred, p.grid, p.numParticles, s);
    }

    void Simulator::kSort(cudaStream_t s, const SimParams& p) {
        ::LaunchSortPairs(m_bufs.d_sortTemp, m_bufs.sortTempBytes,
            m_bufs.d_cellKeys, m_bufs.d_cellKeys_sorted,
            m_bufs.d_indices, m_bufs.d_indices_sorted,
            p.numParticles, s);
    }

    void Simulator::kCellRanges(cudaStream_t s, const SimParams& p) {
        // 修复：先清空单元范围，避免空单元沿用旧帧数据
        CUDA_CHECK(cudaMemsetAsync(m_bufs.d_cellStart, 0xFF, sizeof(uint32_t) * m_numCells, s));
        CUDA_CHECK(cudaMemsetAsync(m_bufs.d_cellEnd, 0xFF, sizeof(uint32_t) * m_numCells, s));

        ::LaunchCellRanges(m_bufs.d_cellStart, m_bufs.d_cellEnd, m_bufs.d_cellKeys_sorted, p.numParticles, m_numCells, s);
    }

    void Simulator::kSolveIter(cudaStream_t s, const SimParams& p) {
        // 从集中参数构造设备侧参数（每次迭代可直接按当前 p 派生）
        sim::DeviceParams dp = sim::MakeDeviceParams(p);

        ::LaunchLambda(m_bufs.d_lambda, m_bufs.d_pos_pred, m_bufs.d_indices_sorted,
            m_bufs.d_cellStart, m_bufs.d_cellEnd,
            dp, p.numParticles, s);

        ::LaunchDeltaApply(m_bufs.d_pos_pred, m_bufs.d_delta, m_bufs.d_lambda, m_bufs.d_indices_sorted,
            m_bufs.d_cellStart, m_bufs.d_cellEnd,
            dp, p.numParticles, s);
    }

    void Simulator::kVelocityAndPost(cudaStream_t s, const SimParams& p) {
        // 1) 先用位置差分得到“约束后的”速度
        ::LaunchVelocity(m_bufs.d_vel, m_bufs.d_pos, m_bufs.d_pos_pred, 1.0f / p.dt, p.numParticles, s);

        // 2) 可选：XSPH 平滑速度（基于本帧邻域）
        if (p.xsph_c > 0.0f && p.numParticles > 0) {
            sim::DeviceParams dp = sim::MakeDeviceParams(p);
            ::LaunchXSPH(
                /*vel_out*/ m_bufs.d_delta,
                /*vel_in */ m_bufs.d_vel,
                /*pos    */ m_bufs.d_pos_pred,
                /*grid   */ m_bufs.d_indices_sorted, m_bufs.d_cellStart, m_bufs.d_cellEnd,
                dp, p.numParticles, s);
            CUDA_CHECK(cudaMemcpyAsync(m_bufs.d_vel, m_bufs.d_delta, sizeof(float4) * p.numParticles,
                cudaMemcpyDeviceToDevice, s));
        }

        // 3) 边界处理应当作用在“最终速度”与位置上，保证反弹/阻尼生效
        ::LaunchBoundary(m_bufs.d_pos_pred, m_bufs.d_vel, p.grid, p.boundaryRestitution, p.numParticles, s);

        // 4) 回收到喷口（如启用）：该核需同时设置新位置与期望初速度
        ::LaunchRecycleToNozzleConst(m_bufs.d_pos, m_bufs.d_pos_pred, m_bufs.d_vel,
            p.grid, p.dt, p.numParticles, /*enabled*/ 1, s);

        // 5) 提交位置
        CUDA_CHECK(cudaMemcpyAsync(m_bufs.d_pos, m_bufs.d_pos_pred, sizeof(float4) * p.numParticles,
            cudaMemcpyDeviceToDevice, s));
    }

    // 固定维度格点布点实现
    void Simulator::seedBoxLattice(uint32_t nx, uint32_t ny, uint32_t nz, float3 origin, float spacing) {
        const uint64_t nreq64 = uint64_t(nx) * uint64_t(ny) * uint64_t(nz);
        uint32_t Nreq = (nreq64 > UINT32_MAX) ? UINT32_MAX : uint32_t(nreq64);

        // 目标 N = min(请求, 容量)
        uint32_t N = Nreq;
        if (N > m_bufs.capacity) {
            m_bufs.allocate(N);
            // 重新初始化 indices
            std::vector<uint32_t> h_idx(N);
            for (uint32_t i = 0; i < N; ++i) h_idx[i] = i;
            CUDA_CHECK(cudaMemcpy(m_bufs.d_indices, h_idx.data(), sizeof(uint32_t) * N, cudaMemcpyHostToDevice));
            m_graphDirty = true;
        }

        // 更新活动粒子数
        m_params.numParticles = N;

        std::vector<float4> h_pos(N);
        uint32_t idx = 0;
        for (uint32_t iz = 0; iz < nz && idx < N; ++iz) {
            for (uint32_t iy = 0; iy < ny && idx < N; ++iy) {
                for (uint32_t ix = 0; ix < nx && idx < N; ++ix) {
                    float x = origin.x + float(ix) * spacing;
                    float y = origin.y + float(iy) * spacing;
                    float z = origin.z + float(iz) * spacing;
                    // 夹在仿真边界内（留半径 margin）
                    x = fminf(fmaxf(x, m_params.grid.mins.x), m_params.grid.maxs.x);
                    y = fminf(fmaxf(y, m_params.grid.mins.y), m_params.grid.maxs.y);
                    z = fminf(fmaxf(z, m_params.grid.mins.z), m_params.grid.maxs.z);
                    h_pos[idx++] = make_float4(x, y, z, 1.0f);
                }
            }
        }
        // 如未刚好填满，用最后一个点补齐
        for (; idx < N; ++idx) h_pos[idx] = (N > 0) ? h_pos[N - 1] : make_float4(origin.x, origin.y, origin.z, 1.0f);

        CUDA_CHECK(cudaMemcpy(m_bufs.d_pos, h_pos.data(), sizeof(float4) * N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m_bufs.d_pos_pred, h_pos.data(), sizeof(float4) * N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(m_bufs.d_vel, 0, sizeof(float4) * N));
    }

    // 通用格点布点：根据仿真域和 spacing 自动分解 nx,ny,nz
    void Simulator::seedBoxLatticeAuto(uint32_t total, float3 origin, float spacing) {
        // 物理尺寸可容纳的最大点数（各轴）
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

        // 近似立方体分解（受上限约束）：ceil 分配以覆盖 total
        uint64_t T = std::max<uint32_t>(1u, total);
        uint32_t nx = std::min<uint32_t>(maxX, (uint32_t)ceil(pow((double)T, 1.0 / 3.0)));
        uint64_t remXY = (T + nx - 1) / nx;
        uint32_t ny = std::min<uint32_t>(maxY, (uint32_t)ceil(sqrt((double)remXY)));
        uint32_t nz = std::min<uint32_t>(maxZ, (uint32_t)ceil((double)remXY / std::max<uint32_t>(1u, ny)));

        // 如果仍不足以覆盖 total，尽力在上限内扩张
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
            std::fprintf(stderr, "[Warn] seedBoxLatticeAuto: Nreq(%u) < total(%u). "
                "Many particles will overlap (tail filled by last point). "
                "Consider decreasing smoothingRadius or using adaptive spacing.\n", Nreq, total);
        }

        // 原点拉回至边界内留出一层 margin
        float3 margin = make_float3(spacing * 0.25f, spacing * 0.25f, spacing * 0.25f);
        origin.x = fminf(fmaxf(origin.x, m_params.grid.mins.x + margin.x), m_params.grid.maxs.x - margin.x);
        origin.y = fminf(fmaxf(origin.y, m_params.grid.mins.y + margin.y), m_params.grid.maxs.y - margin.y);
        origin.z = fminf(fmaxf(origin.z, m_params.grid.mins.z + margin.z), m_params.grid.maxs.z - margin.z);

        seedBoxLattice(nx, ny, nz, origin, spacing);
    }

    bool Simulator::importPosPredFromD3D12(void* sharedHandleWin32, size_t bytes) {
        if (!sharedHandleWin32 || bytes == 0) return false;

        // 释放旧映射（若存在）
        if (m_extPosPred) {
            cudaDestroyExternalMemory(m_extPosPred);
            m_extPosPred = nullptr;
            m_bufs.detachExternalPosPred();
        }

        cudaExternalMemoryHandleDesc memDesc{};
        memDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
        memDesc.handle.win32.handle = sharedHandleWin32;
        memDesc.size = bytes;
        memDesc.flags = cudaExternalMemoryDedicated; // committed resource

        CUDA_CHECK(cudaImportExternalMemory(&m_extPosPred, &memDesc));

        cudaExternalMemoryBufferDesc bufDesc{};
        bufDesc.offset = 0;
        bufDesc.size = bytes;
        bufDesc.flags = 0;

        void* devPtr = nullptr;
        CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&devPtr, m_extPosPred, &bufDesc));

        // 绑定为 d_pos_pred（释放原自有 d_pos_pred 以避免泄漏）
        m_bufs.bindExternalPosPred(reinterpret_cast<float4*>(devPtr));
        return true;
    }
    bool Simulator::computeStats(SimStats& out, uint32_t sampleStride) const {
        if (m_params.numParticles == 0 || m_bufs.capacity == 0) { out = {}; return true; }

        // 1) 先跑基于网格的统计
        double avgN_grid = 0.0, avgV = 0.0, avgRRel_grid = 0.0, avgR_grid = 0.0;
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
            &avgN_grid, &avgV, &avgRRel_grid, &avgR_grid,
            m_stream);
        if (!ok_grid) return false;

        // 默认以 grid 结果回填输出
        out.N = m_params.numParticles;
        out.avgNeighbors = avgN_grid;
        out.avgSpeed = avgV;
        out.avgRhoRel = avgRRel_grid;
        out.avgRho = avgR_grid;

        // 2) 跑一小批暴力法，对比差异（代价有限，便于诊断）
        double avgN_bf = 0.0, avgV_bf = 0.0, avgRRel_bf = 0.0, avgR_bf = 0.0;
        const uint32_t kMaxISamples = 2048; // 限制参与暴力法的采样粒子数，控制开销
        bool ok_bf = LaunchComputeStatsBruteforce(
            m_bufs.d_pos_pred,
            m_bufs.d_vel,
            m_params.kernel,
            m_params.particleMass,
            m_params.numParticles,
            stride,
            kMaxISamples,
            &avgN_bf, &avgV_bf, &avgRRel_bf, &avgR_bf,
            m_stream);

        if (!ok_bf) return true; // 暴力统计失败时，不中断，直接返回 grid 结果

        // 3) 对比并判定异常
        const bool hasNeighborCap = (m_params.maxNeighbors > 0);
        const double capN = hasNeighborCap ? double(m_params.maxNeighbors) : std::numeric_limits<double>::infinity();
        const double nearCapRatio = 0.9; // 90% 视为接近上限
        const bool nearCap = hasNeighborCap && (avgN_grid >= nearCapRatio * capN);

        // grid 与暴力差距（邻居或密度）阈值
        const double nAbsThresh = 2.0;      // 绝对差大于 2 视为可疑
        const double nRelThresh = 0.5;      // grid 结果比暴力法低于 50%
        const bool nSeverelyUndercount =
            (avgN_bf > 0.0) && ((avgN_grid + nAbsThresh) < avgN_bf * (1.0 - nRelThresh));

        // 4) 仅在发现异常时打印诊断（同帧节流）
        static uint64_t s_lastDiagFrame = UINT64_MAX;
        const bool shouldDiag = (nSeverelyUndercount || nearCap);
        if (shouldDiag && s_lastDiagFrame != m_frameIndex) {
            s_lastDiagFrame = m_frameIndex;

            // 4.1 基础参数与建议
            const double h = double(m_params.kernel.h);
            const double cell = double(m_params.grid.cellSize);
            const double ratio_cell_h = (h > 0.0) ? (cell / h) : 0.0;

            std::fprintf(stderr,
                "[Diag] Frame=%llu | N=%u | h=%.6g | cell=%.6g (cell/h=%.3f) | dim=(%d,%d,%d) numCells=%u\n",
                (unsigned long long)m_frameIndex, m_params.numParticles, h, cell, ratio_cell_h,
                m_params.grid.dim.x, m_params.grid.dim.y, m_params.grid.dim.z, m_numCells);

            std::fprintf(stderr,
                "[Diag] Neighbors: grid=%.3f, brute=%.3f | RhoRel: grid=%.3f, brute=%.3f | maxNeighbors=%d%s\n",
                avgN_grid, avgN_bf, avgRRel_grid, avgRRel_bf, m_params.maxNeighbors,
                (nearCap ? " (near cap -> risk of truncation)" : ""));

            if (ratio_cell_h < 0.9) {
                std::fprintf(stderr, "[Hint] cellSize 明显小于 h：请设为 ~[1.0, 1.5]h，过小会导致需要更大邻接层数或漏邻居。\n");
            }
            else if (ratio_cell_h > 2.0) {
                std::fprintf(stderr, "[Hint] cellSize 明显大于 h：请设为 ~[1.0, 1.5]h，过大将导致单元过大、邻居暴涨与排序压力。\n");
            }
            if (nearCap) {
                std::fprintf(stderr, "[Hint] avgNeighbors_grid 接近上限 maxNeighbors=%d：建议临时设为 0（无限）或调大，以排除截断偏差。\n", m_params.maxNeighbors);
            }
            if (nSeverelyUndercount) {
                std::fprintf(stderr, "[Hint] grid 邻居显著小于暴力法：排查 cellSize/h 是否匹配、邻域遍历是否完整（不要因达到上限而提前停止遍历）。\n");
            }

            // 4.2 回读单元范围，做占用直方（帮助判断“团聚”与“漏邻居”）
            CUDA_CHECK(cudaStreamSynchronize(m_stream));
            std::vector<uint32_t> h_start(m_numCells), h_end(m_numCells);
            CUDA_CHECK(cudaMemcpy(h_start.data(), m_bufs.d_cellStart, sizeof(uint32_t) * m_numCells, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_end.data(), m_bufs.d_cellEnd, sizeof(uint32_t) * m_numCells, cudaMemcpyDeviceToHost));

            const uint32_t EMPTY = 0xFFFFFFFFu;
            uint64_t nonEmpty = 0, empty = 0;
            uint64_t sumCnt = 0;
            uint64_t sumCnt2 = 0;
            uint32_t minCnt = UINT32_MAX, maxCnt = 0;
            uint32_t maxIdx = 0;
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
                "[Diag] Cell occupancy: empty=%llu (%.2f%%), non-empty=%llu | per-cell cnt: min=%u, avg=%.2f, max=%u (cell=(%d,%d,%d), frac=%.2f%%) | std=%.2f\n",
                (unsigned long long)empty, 100.0 * double(empty) / double(m_numCells),
                (unsigned long long)nonEmpty,
                (unsigned)(nonEmpty ? minCnt : 0u), meanCnt, (unsigned)maxCnt,
                cMax.x, cMax.y, cMax.z, 100.0 * fracMaxCell, stdCnt);

            if (fracMaxCell > 0.05) {
                std::fprintf(stderr, "[Warn] 单一网格单元承载了 %.2f%% 的粒子，出现明显团聚。请检查喷口尺度/域尺度、h 与发射速率是否匹配。\n", 100.0 * fracMaxCell);
            }
        }

        return true;
    }
    bool Simulator::computeStatsBruteforce(SimStats& out, uint32_t sampleStride, uint32_t maxISamples) const {
        if (m_params.numParticles == 0) { out = {}; return true; }
        double avgN = 0.0, avgV = 0.0, avgRRel = 0.0, avgR = 0.0;
        bool ok = LaunchComputeStatsBruteforce(
            m_bufs.d_pos_pred,
            m_bufs.d_vel,
            m_params.kernel,
            m_params.particleMass,
            m_params.numParticles,
            (sampleStride == 0 ? 1u : sampleStride),
            maxISamples,
            &avgN, &avgV, &avgRRel, &avgR,
            m_stream);
        if (!ok) return false;
        out.N = m_params.numParticles;
        out.avgNeighbors = avgN;
        out.avgSpeed = avgV;
        out.avgRhoRel = avgRRel;
        out.avgRho = avgR;
        return true;
    }

    bool Simulator::computeCenterOfMass(float3& outCom, uint32_t sampleStride) const {
        outCom = make_float3(0, 0, 0);
        const uint32_t N = m_params.numParticles;
        if (N == 0) return true;
        std::vector<float4> h_pos(N);
        // 从共享缓冲导入也可见；此处直接从设备拷贝
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
} // namespace sim