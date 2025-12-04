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

// Build per-cell ranges [start, end) for a sorted key array.
// Preconditions:
// - keysSorted contains cell keys in non-decreasing order.
// - cellStart is initialized to 0xFFFFFFFF, cellEnd to 0 (caller).
// Postconditions:
// - For a cell with elements: start[idx] is the first position, end[idx] is one past the last.
// - For an empty cell: start[idx] stays 0xFFFFFFFF, end[idx] stays 0.
__global__ void KBuildRanges(const uint32_t* __restrict__ keysSorted,
                             uint32_t N,
                             uint32_t numCells,
                             uint32_t* __restrict__ start,
                             uint32_t* __restrict__ end) {
    // Grid-stride loop to cover arbitrary N robustly.
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        const uint32_t k = keysSorted[i];
        if (k >= numCells) {
            // Defensive: ignore out-of-range keys.
            continue;
        }

        if (i == 0) {
            // First element opens its cell at position 0.
            start[k] = 0;
        } else {
            const uint32_t kPrev = keysSorted[i - 1];
            if (k != kPrev) {
                // Cell boundary: close previous cell at i, open current cell at i.
                if (kPrev < numCells) end[kPrev] = i;
                start[k] = i;
            }
        }

        if (i == N - 1) {
            // Last element closes its cell at N.
            end[k] = N;
        }
    }
}

} // namespace

extern "C" void LaunchCellRanges(uint32_t* cellStart,
                                 uint32_t* cellEnd,
                                 const uint32_t* keysSorted,
                                 uint32_t N,
                                 uint32_t numCells,
                                 cudaStream_t s) {
    // Initialize to "empty": start=0xFFFFFFFF, end=0.
    CUDA_CHECK(cudaMemsetAsync(cellStart, 0xFF, sizeof(uint32_t) * numCells, s));
    CUDA_CHECK(cudaMemsetAsync(cellEnd,   0x00, sizeof(uint32_t) * numCells, s));

    if (N == 0 || numCells == 0) {
        // Nothing to do.
        return;
    }

    const int BS = 256;
    const int maxBlocks = 1024; // reasonable cap to avoid oversubscription
    int blocks = (int)((N + BS - 1) / BS);
    if (blocks > maxBlocks) blocks = maxBlocks;

    // Launch builder.
    KBuildRanges<<<blocks, BS, 0, s>>>(keysSorted, N, numCells, cellStart, cellEnd);

    // Robustness: check for launch errors.
    CUDA_CHECK(cudaGetLastError());
}