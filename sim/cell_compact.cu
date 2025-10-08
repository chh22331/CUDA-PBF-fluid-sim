#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

// 输入：keysSorted[N]
// 输出：uniqueKeys[M]（升序，非空 cell 的 key），offsets[M+1]（每段起止），*compactCount=M
// 说明：单线程 O(N) 构建，避免对临时/前缀和的需求，便于纳入 CUDA Graph。
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

    // 写首段
    uint32_t segCount = 0u;           // 段计数（已写入的最后一段下标）
    uint32_t prev = keysSorted[0];
    uniqueKeys[0] = prev;
    offsets[0] = 0u;

    // 从第 1 个元素开始扫描，遇到切换即写段头
    for (uint32_t i = 1u; i < N; ++i) {
        const uint32_t k = keysSorted[i];
        if (k != prev) {
            ++segCount;
            uniqueKeys[segCount] = k;
            offsets[segCount] = i; // 段 segCount 的起点是 i
            prev = k;
        }
    }

    // 末尾封口：offsets[M] = N
    offsets[segCount + 1u] = N;

    // M = 段数 = segCount + 1
    *compactCount = (segCount + 1u);
}

extern "C" void LaunchCellRangesCompact(
    uint32_t* d_uniqueKeys,
    uint32_t* d_offsets,
    uint32_t* d_compactCount,
    const uint32_t* d_keysSorted,
    uint32_t N,
    cudaStream_t s)
{
    dim3 grid(1), block(1);
    KBuildCompactSequential << <grid, block, 0, s >> > (
        d_keysSorted, N, d_uniqueKeys, d_offsets, d_compactCount);
}