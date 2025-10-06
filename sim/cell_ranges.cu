#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); abort(); } } while(0)
#endif

namespace {
    // 假设 keysSorted 已按 cell key 升序
    // 写入每个 cell 的 [start, end)，空 cell: start=0xFFFFFFFF, end=0
    __global__ void KBuildRanges(const uint32_t* keysSorted, uint32_t N,
        uint32_t* start, uint32_t* end) {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;

        uint32_t k = keysSorted[i];

        if (i == 0) {
            // 第一个元素所在 cell 起始为 0
            start[k] = 0;
        }
        else {
            uint32_t kPrev = keysSorted[i - 1];
            if (k != kPrev) {
                // 发生 cell 变化：关闭前一个 cell，开启当前 cell
                end[kPrev] = i;
                start[k] = i;
            }
        }
        if (i == N - 1) {
            // 最后一个元素关闭其 cell 的 end=N
            end[k] = N;
        }
    }
}

extern "C" void LaunchCellRanges(uint32_t* cellStart, uint32_t* cellEnd,
    const uint32_t* keysSorted, uint32_t N,
    uint32_t numCells, cudaStream_t s) {
    // 初始化：start=0xFFFFFFFF, end=0
    cudaMemsetAsync(cellStart, 0xFF, sizeof(uint32_t) * numCells, s);
    cudaMemsetAsync(cellEnd, 0x00, sizeof(uint32_t) * numCells, s);

    const int BS = 256;
    dim3 b(BS), g((N + BS - 1) / BS);
    KBuildRanges << <g, b, 0, s >> > (keysSorted, N, cellStart, cellEnd);

    // 健壮性：检查 launch 错误
    CUDA_CHECK(cudaGetLastError());
}