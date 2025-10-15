#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cub/cub.cuh>

// 可选：调试回退
//#define USE_SEQUENTIAL_FALLBACK 0
//#define USE_TILED_FALLBACK 0

// ==== 原始顺序版本（保留以便调试） ====
__global__ void KBuildCompactSequential(
    const uint32_t* keysSorted,
    uint32_t N,
    uint32_t* uniqueKeys,
    uint32_t* offsets,
    uint32_t* compactCount)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (N == 0u) {
        if (offsets) offsets[0] = 0u;
        if (compactCount) *compactCount = 0u;
        return;
    }
    uint32_t segCount = 0u;
    uint32_t prev = keysSorted[0];
    uniqueKeys[0] = prev;
    offsets[0] = 0u;
    for (uint32_t i = 1u; i < N; ++i) {
        uint32_t k = keysSorted[i];
        if (k != prev) {
            ++segCount;
            uniqueKeys[segCount] = k;
            offsets[segCount] = i;
            prev = k;
        }
    }
    offsets[segCount + 1u] = N;
    *compactCount = segCount + 1u;
}

// ==== 第一阶段单块分块版本（保留回退） ====
__global__ void KBuildCompactTiled(
    const uint32_t* __restrict__ keysSorted,
    uint32_t N,
    uint32_t* __restrict__ uniqueKeys,
    uint32_t* __restrict__ offsets,
    uint32_t* __restrict__ compactCount)
{
    if (blockIdx.x != 0) return;
    constexpr int TILE = 256;
    if (N == 0u) { offsets[0] = 0u; *compactCount = 0u; return; }
    if (N == 1u) {
        uniqueKeys[0] = keysSorted[0];
        offsets[0] = 0u; offsets[1] = 1u; *compactCount = 1u; return;
    }
    uint32_t segCount = 0u;
    if (threadIdx.x == 0) {
        uniqueKeys[0] = keysSorted[0];
        offsets[0] = 0u;
        segCount = 1u;
    }
    __syncthreads();
    for (uint32_t base = 1u; base < N; base += TILE) {
        uint32_t end = base + TILE; if (end > N) end = N;
        uint32_t i = base + threadIdx.x;
        __shared__ uint32_t sFlags[TILE];
        __shared__ uint32_t sRanks[TILE];
        uint32_t flag = 0u;
        if (i < end) flag = (keysSorted[i] != keysSorted[i - 1]) ? 1u : 0u;
        sFlags[threadIdx.x] = flag;
        __syncthreads();
        if (threadIdx.x == 0) {
            uint32_t running = 0u;
            for (int t = 0; t < TILE; ++t) {
                sRanks[t] = running;
                running += sFlags[t];
            }
            sRanks[TILE - 1] = running; // 存放本 tile 新段数
        }
        __syncthreads();
        uint32_t newSeg = sRanks[TILE - 1];
        if (flag) {
            uint32_t outIdx = segCount + sRanks[threadIdx.x];
            uniqueKeys[outIdx] = keysSorted[i];
            offsets[outIdx] = i;
        }
        __syncthreads();
        if (threadIdx.x == 0) segCount += newSeg;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        offsets[segCount] = N;
        *compactCount = segCount;
    }
}

// =====================================================
// 最终完全并行版本
// Phase1: KMarkAndCountWarp (标记段起点 + 每块段数)
// Phase2: CUB DeviceScan ExclusiveSum (块段数前缀和 -> blockOffsets)
// Phase3: KScatterSegmentsWarp (warp ballot + 局部 rank + 全局写入)
// Scratch:
//   g_flags (uint8_t[N])
//   g_blockCounts (numBlocks)
//   g_blockOffsets (numBlocks)
//   g_scanTemp (CUB 临时)
// =====================================================

static uint8_t*  g_flags         = nullptr;
static uint32_t* g_blockCounts   = nullptr;
static uint32_t* g_blockOffsets  = nullptr;
static void*     g_scanTemp      = nullptr;
static size_t    g_scanTempBytes = 0;
static uint32_t  g_scratchCapacityElems = 0;
static uint32_t  g_scratchCapacityBlocks = 0;

extern "C" bool EnsureCellCompactScratch(uint32_t N, uint32_t threadsPerBlock = 256)
{
    if (threadsPerBlock == 0) threadsPerBlock = 256;
    uint32_t numBlocks = (N + threadsPerBlock - 1u) / threadsPerBlock;
    if (numBlocks == 0) numBlocks = 1;

    bool ok = true;

    // 重新分配元素级 flags
    if (N > g_scratchCapacityElems) {
        if (g_flags) cudaFree(g_flags);
        if (cudaMalloc((void**)&g_flags, sizeof(uint8_t) * N) != cudaSuccess) ok = false;
        g_scratchCapacityElems = N;
    }

    // 重新分配块级缓冲
    if (numBlocks > g_scratchCapacityBlocks) {
        if (g_blockCounts)  cudaFree(g_blockCounts);
        if (g_blockOffsets) cudaFree(g_blockOffsets);
        if (cudaMalloc((void**)&g_blockCounts,  sizeof(uint32_t) * numBlocks) != cudaSuccess) ok = false;
        if (cudaMalloc((void**)&g_blockOffsets, sizeof(uint32_t) * numBlocks) != cudaSuccess) ok = false;
        g_scratchCapacityBlocks = numBlocks;
    }

    if (!ok) return false;

    // 预查询 CUB 扫描临时
    size_t needed = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, needed, g_blockCounts, g_blockOffsets, numBlocks);
    if (needed > g_scanTempBytes) {
        if (g_scanTemp) cudaFree(g_scanTemp);
        if (cudaMalloc(&g_scanTemp, needed) != cudaSuccess) return false;
        g_scanTempBytes = needed;
    }
    return true;
}

// Phase1: 标记 + 统计块段数 (无全局原子，块内 warp 归约)
__global__ void KMarkAndCountWarp(
    const uint32_t* __restrict__ keysSorted,
    uint32_t N,
    uint8_t*  __restrict__ flags,
    uint32_t* __restrict__ blockCounts)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t flag = 0u;
    if (gid < N) {
        if (gid == 0) flag = 1u;
        else flag = (keysSorted[gid] != keysSorted[gid - 1]) ? 1u : 0u;
        flags[gid] = static_cast<uint8_t>(flag);
    }

    // warp ballot 汇总
    unsigned mask = __ballot_sync(0xFFFFFFFFu, (gid < N && flag));
    uint32_t warpCount = __popc(mask);

    // 每 warp 写共享内存，再由 0 号线程归约
    __shared__ uint32_t sWarpCounts[32]; // 支持最多 32 warps (blockDim<=1024)
    int warpId = threadIdx.x >> 5;
    int lane   = threadIdx.x & 31;
    if (lane == 0) sWarpCounts[warpId] = warpCount;
    __syncthreads();

    if (threadIdx.x == 0) {
        uint32_t sum = 0u;
        int warps = (blockDim.x + 31) >> 5;
        for (int w = 0; w < warps; ++w) sum += sWarpCounts[w];
        blockCounts[blockIdx.x] = sum;
    }
}

// Phase3: 根据块基址 + warp 局部 rank 写段起点与 uniqueKeys
__global__ void KScatterSegmentsWarp(
    const uint32_t* __restrict__ keysSorted,
    uint32_t N,
    const uint8_t* __restrict__ flags,
    const uint32_t* __restrict__ blockOffsets,
    const uint32_t* __restrict__ blockCounts,
    uint32_t* __restrict__ uniqueKeys,
    uint32_t* __restrict__ offsets,
    uint32_t* __restrict__ compactCount)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t f = (gid < N) ? (uint32_t)flags[gid] : 0u;

    // warp 内 rank 利用 ballot + laneMask
    unsigned ballot = __ballot_sync(0xFFFFFFFFu, f);
    int lane = threadIdx.x & 31;
    uint32_t laneMaskLess = (1u << lane) - 1u;
    uint32_t localRank = f ? __popc(ballot & laneMaskLess) : 0u;
    uint32_t warpCount = __popc(ballot);

    // 共享内存存各 warp 段计数
    __shared__ uint32_t sWarpBase[32];
    int warpId = threadIdx.x >> 5;
    if (lane == 0) sWarpBase[warpId] = warpCount;
    __syncthreads();

    // 0 号线程做 warp 级前缀，写回 sWarpBase 为每 warp 基址
    if (threadIdx.x == 0) {
        uint32_t running = 0u;
        int warps = (blockDim.x + 31) >> 5;
        for (int w = 0; w < warps; ++w) {
            uint32_t c = sWarpBase[w];
            sWarpBase[w] = running;
            running += c;
        }
    }
    __syncthreads();

    // 计算全局段索引
    if (f) {
        uint32_t blockBase = blockOffsets[blockIdx.x];
        uint32_t warpBase  = sWarpBase[warpId];
        uint32_t globalIdx = blockBase + warpBase + localRank;
        uniqueKeys[globalIdx] = keysSorted[gid];
        offsets[globalIdx]    = gid;
    }

    // 末块写封口与段总数
    if (blockIdx.x == gridDim.x - 1 && threadIdx.x == 0) {
        uint32_t lastBlock = blockIdx.x;
        uint32_t totalSegments = blockOffsets[lastBlock] + blockCounts[lastBlock];
        offsets[totalSegments] = N;
        *compactCount = totalSegments;
    }
}

// === 启动封装：默认使用最终并行版本 ===
extern "C" void LaunchCellRangesCompact(
    uint32_t* d_uniqueKeys,
    uint32_t* d_offsets,
    uint32_t* d_compactCount,
    const uint32_t* d_keysSorted,
    uint32_t N,
    cudaStream_t s)
{
#if defined(USE_SEQUENTIAL_FALLBACK)
    dim3 g(1), b(1);
    KBuildCompactSequential<<<g, b, 0, s>>>(d_keysSorted, N, d_uniqueKeys, d_offsets, d_compactCount);
    return;
#elif defined(USE_TILED_FALLBACK)
    dim3 g(1), b(256);
    KBuildCompactTiled<<<g, b, 0, s>>>(d_keysSorted, N, d_uniqueKeys, d_offsets, d_compactCount);
    return;
#else
    if (N == 0u) {
        cudaMemsetAsync(d_offsets, 0, sizeof(uint32_t), s);
        cudaMemsetAsync(d_compactCount, 0, sizeof(uint32_t), s);
        return;
    }
    const uint32_t blockDimThreads = 256;
    uint32_t numBlocks = (N + blockDimThreads - 1u) / blockDimThreads;
    if (numBlocks == 0) numBlocks = 1;

    // Graph 捕获前请先调用 EnsureCellCompactScratch(N)
    if (!EnsureCellCompactScratch(N, blockDimThreads)) {
        // 安全回退
        dim3 g(1), b(1);
        KBuildCompactSequential<<<g, b, 0, s>>>(d_keysSorted, N, d_uniqueKeys, d_offsets, d_compactCount);
        return;
    }

    // Phase1: 标记 + 统计
    KMarkAndCountWarp<<<numBlocks, blockDimThreads, 0, s>>>(
        d_keysSorted, N, g_flags, g_blockCounts);

    // Phase2: 块段数前缀和
    cub::DeviceScan::ExclusiveSum(
        g_scanTemp, g_scanTempBytes,
        g_blockCounts, g_blockOffsets,
        numBlocks, s);

    // Phase3: 散射写段与封口
    KScatterSegmentsWarp<<<numBlocks, blockDimThreads, 0, s>>>(
        d_keysSorted, N,
        g_flags,
        g_blockOffsets,
        g_blockCounts,
        d_uniqueKeys,
        d_offsets,
        d_compactCount);
#endif
}