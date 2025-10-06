#pragma once
#include <cuda_runtime.h>

namespace sim {
    struct EmitParams {
        float3 nozzlePos;
        float3 nozzleDir;
        float  nozzleRadius;
        float  nozzleSpeed;
        float  recycleY; // 回收阈值（典型为 grid.mins.y + eps）
    };
}

// C 接口，便于 .cpp 调用与链接
extern "C" void SetEmitParamsAsync(const sim::EmitParams* h, cudaStream_t s);
extern "C" void* GetRecycleKernelPtr(); // 获取回收内核函数指针（用于定位图节点）