#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

namespace {
    __global__ void KBuildRanges(const uint32_t* keysSorted, uint32_t N, uint32_t* start, uint32_t* end, uint32_t numCells) {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i == 0) {
            for (uint32_t c = 0; c < numCells; ++c) { start[c] = 0xFFFFFFFFu; end[c] = 0u; }
        }
        __syncthreads();

        if (i >= N) return;
        uint32_t k = keysSorted[i];
        uint32_t kPrev = (i > 0) ? keysSorted[i - 1] : 0xFFFFFFFFu;
        if (i == 0 || k != kPrev) start[k] = i;
        if (i == N - 1) end[k] = N;
        else {
            uint32_t kNext = keysSorted[i + 1];
            if (k != kNext) end[k] = i + 1;
        }
    }
}

extern "C" void LaunchCellRanges(uint32_t* cellStart, uint32_t* cellEnd, const uint32_t* keysSorted, uint32_t N, uint32_t numCells, cudaStream_t s) {
    const int BS = 256;
    dim3 b(BS), g((N + BS - 1) / BS);
    KBuildRanges<<<g, b, 0, s>>>(keysSorted, N, cellStart, cellEnd, numCells);
}