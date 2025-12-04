#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cub/cub.cuh>

// Optional sequential fallback used only when scratch allocation fails.
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

// =====================================================
// fully parallel path:
// Phase 1: KMarkAndCountWarp  -> marks segment starts & counts per block
// Phase 2: CUB ExclusiveScan  -> scans per-block counts into global offsets
// Phase 3: KScatterSegmentsWarp -> writes unique keys + segment offsets
// Scratch buffers:
//   g_flags        (uint8_t[N])
//   g_blockCounts  (uint32_t[numBlocks])
//   g_blockOffsets (uint32_t[numBlocks])
//   g_scanTemp     (dynamic CUB temporary)
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

    // Reallocate element-level flag buffer if required.
    if (N > g_scratchCapacityElems) {
        if (g_flags) cudaFree(g_flags);
        if (cudaMalloc((void**)&g_flags, sizeof(uint8_t) * N) != cudaSuccess) ok = false;
        g_scratchCapacityElems = N;
    }

    // Reallocate per-block counters if current capacity is insufficient.
    if (numBlocks > g_scratchCapacityBlocks) {
        if (g_blockCounts)  cudaFree(g_blockCounts);
        if (g_blockOffsets) cudaFree(g_blockOffsets);
        if (cudaMalloc((void**)&g_blockCounts,  sizeof(uint32_t) * numBlocks) != cudaSuccess) ok = false;
        if (cudaMalloc((void**)&g_blockOffsets, sizeof(uint32_t) * numBlocks) != cudaSuccess) ok = false;
        g_scratchCapacityBlocks = numBlocks;
    }

    if (!ok) return false;

    // Query CUB for temporary storage requirements.
    size_t needed = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, needed, g_blockCounts, g_blockOffsets, numBlocks);
    if (needed > g_scanTempBytes) {
        if (g_scanTemp) cudaFree(g_scanTemp);
        if (cudaMalloc(&g_scanTemp, needed) != cudaSuccess) return false;
        g_scanTempBytes = needed;
    }
    return true;
}

// Phase 1: mark segment starts & count per block without global atomics.
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

    // Aggregate per-warp counts via ballot.
    unsigned mask = __ballot_sync(0xFFFFFFFFu, (gid < N && flag));
    uint32_t warpCount = __popc(mask);

    // Stash counts per warp in shared memory for block-level reduction.
    __shared__ uint32_t sWarpCounts[32]; // Up to 32 warps (blockDim <= 1024).
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

// Phase 3: use block offsets + warp-local ranks to scatter out segments.
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

    // Compute warp-local rank via ballot + lane mask.
    unsigned ballot = __ballot_sync(0xFFFFFFFFu, f);
    int lane = threadIdx.x & 31;
    uint32_t laneMaskLess = (1u << lane) - 1u;
    uint32_t localRank = f ? __popc(ballot & laneMaskLess) : 0u;
    uint32_t warpCount = __popc(ballot);

    // Cache warp segment counts.
    __shared__ uint32_t sWarpBase[32];
    int warpId = threadIdx.x >> 5;
    if (lane == 0) sWarpBase[warpId] = warpCount;
    __syncthreads();

    // Thread 0 builds an exclusive prefix over warp counts.
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

    // Emit unique keys and offsets using the scanned bases.
    if (f) {
        uint32_t blockBase = blockOffsets[blockIdx.x];
        uint32_t warpBase  = sWarpBase[warpId];
        uint32_t globalIdx = blockBase + warpBase + localRank;
        uniqueKeys[globalIdx] = keysSorted[gid];
        offsets[globalIdx]    = gid;
    }

    // Last block writes the terminator and total segment count.
    if (blockIdx.x == gridDim.x - 1 && threadIdx.x == 0) {
        uint32_t lastBlock = blockIdx.x;
        uint32_t totalSegments = blockOffsets[lastBlock] + blockCounts[lastBlock];
        offsets[totalSegments] = N;
        *compactCount = totalSegments;
    }
}

// Single public launch entry; always prefers the warp-optimized path.
extern "C" void LaunchCellRangesCompact(
    uint32_t* d_uniqueKeys,
    uint32_t* d_offsets,
    uint32_t* d_compactCount,
    const uint32_t* d_keysSorted,
    uint32_t N,
    cudaStream_t s)
{
    if (N == 0u) {
        cudaMemsetAsync(d_offsets, 0, sizeof(uint32_t), s);
        cudaMemsetAsync(d_compactCount, 0, sizeof(uint32_t), s);
        return;
    }
    const uint32_t blockDimThreads = 256;
    uint32_t numBlocks = (N + blockDimThreads - 1u) / blockDimThreads;
    if (numBlocks == 0) numBlocks = 1;

    // Capture-time callers must pre-size scratch via EnsureCellCompactScratch.
    if (!EnsureCellCompactScratch(N, blockDimThreads)) {
        // Guaranteed fallback keeps simulation functional albeit slower.
        dim3 g(1), b(1);
        KBuildCompactSequential<<<g, b, 0, s>>>(d_keysSorted, N, d_uniqueKeys, d_offsets, d_compactCount);
        return;
    }

    // Phase 1: mark segment starts & count per block.
    KMarkAndCountWarp<<<numBlocks, blockDimThreads, 0, s>>>(
        d_keysSorted, N, g_flags, g_blockCounts);

    // Phase 2: prefix-sum block segment counts.
    cub::DeviceScan::ExclusiveSum(
        g_scanTemp, g_scanTempBytes,
        g_blockCounts, g_blockOffsets,
        numBlocks, s);

    // Phase 3: scatter segments and finish with a terminator.
    KScatterSegmentsWarp<<<numBlocks, blockDimThreads, 0, s>>>(
        d_keysSorted, N,
        g_flags,
        g_blockOffsets,
        g_blockCounts,
        d_uniqueKeys,
        d_offsets,
        d_compactCount);
}
