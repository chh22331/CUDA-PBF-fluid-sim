#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "parameters.h"
#include "cuda_vec_math.cuh"

__device__ __forceinline__ uint32_t linearIndex(int3 c, int3 dim) {
    c.x = max(0, min(c.x, dim.x - 1));
    c.y = max(0, min(c.y, dim.y - 1));
    c.z = max(0, min(c.z, dim.z - 1));
    return (uint32_t)((c.z * dim.y + c.y) * dim.x + c.x);
}

namespace {
    __global__ void KHash(uint32_t* keys, uint32_t* indices, const float4* pos_pred, sim::GridBounds grid, uint32_t N) {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;
        float3 p = to_float3(pos_pred[i]);
        float3 rel = make_float3(
            (p.x - grid.mins.x) / grid.cellSize,
            (p.y - grid.mins.y) / grid.cellSize,
            (p.z - grid.mins.z) / grid.cellSize
        );
        int3 c = make_int3(floorf(rel.x), floorf(rel.y), floorf(rel.z));
        uint32_t key = linearIndex(c, grid.dim);
        keys[i] = key;
        indices[i] = i;
    }
}

extern "C" void LaunchHashKeys(uint32_t* keys, uint32_t* indices, const float4* pos, sim::GridBounds grid, uint32_t N, cudaStream_t s) {
    const int BS = 256;
    dim3 block(BS), gridDim((N + BS - 1) / BS);
    KHash<<<gridDim, block, 0, s>>>(keys, indices, pos, grid, N);
}