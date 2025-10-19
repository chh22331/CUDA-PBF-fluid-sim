#include "precision_traits.cuh"
#include "device_buffers.cuh"
#include "parameters.h"

extern "C" {

    // 占位：后续将加入半精只读加载优化（当前直接调用原数据，不做任何修改）
    __global__ void KHashKeysMP(uint32_t* keys, uint32_t* indices,
        const float4* pos_pred_fp32,
        const sim::Half4* pos_pred_h4,
        sim::GridBounds grid,
        uint32_t N)
    {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;
        float4 p = sim::PrecisionTraits::loadPosPred(pos_pred_fp32, pos_pred_h4, i);
        // TODO: 原 Hash 函数逻辑移植（此处暂用伪实现）
        uint32_t key = 0u; // placeholder
        keys[i] = key;
        indices[i] = i;
    }

    void LaunchHashKeysMP(uint32_t* keys, uint32_t* indices,
        const float4* pos_pred,
        const sim::Half4* pos_pred_h4,
        sim::GridBounds grid,
        uint32_t N,
        cudaStream_t s)
    {
        if (N == 0) return;
        uint32_t threads = 256;
        uint32_t blocks = (N + threads - 1) / threads;
        KHashKeysMP << <blocks, threads, 0, s >> > (keys, indices, pos_pred, pos_pred_h4, grid, N);
    }

} // extern "C"