#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <limits>
#include <stdint.h>

// 输入：按 key 排序后的 keysSorted（此处直接用 d_cellKeys 排序后数组）
// 输出：每 cell 的起止索引 [start,end)；未出现的 cell 设为 start=end
namespace {
    __global__ void KInit(uint32_t* start, uint32_t* end, uint32_t numCells) {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= numCells) return;
        start[i] = 0xFFFFFFFFu; end[i] = 0u;
    }
    __global__ void KMark(const uint32_t* keysSorted, uint32_t N, uint32_t* start, uint32_t* end) {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= N) return;
        uint32_t k = keysSorted[i];
        atomicMin(&start[k], i);
        atomicMax(&end[k], i + 1);
    }
}

extern "C" void LaunchCellRanges(uint32_t* cellStart, uint32_t* cellEnd, const uint32_t* keysSorted, uint32_t N, uint32_t numCells, cudaStream_t s) {
    const int BS = 256;
    dim3 b(BS), g0((numCells + BS - 1) / BS), g1((N + BS - 1) / BS);
    KInit << <g0, b, 0, s >> > (cellStart, cellEnd, numCells);
    KMark << <g1, b, 0, s >> > (keysSorted, N, cellStart, cellEnd);
}