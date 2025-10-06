#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace sim {

// 在 NVCC 编译环境下（包含 .cu 的 host/device 两个 pass），只生成设备端版本，
// 彻底避免主机符号在多个 .cu 的 host pass 中多重定义。
#if defined(__CUDACC__)

// Device-only 内联：供 __global__/__device__ 代码使用
__device__ __forceinline__
bool inBounds(int3 c, int3 dim) {
    return (c.x >= 0 && c.x < dim.x &&
            c.y >= 0 && c.y < dim.y &&
            c.z >= 0 && c.z < dim.z);
}

__device__ __forceinline__
uint32_t linIdx(int3 c, int3 dim) {
    return static_cast<uint32_t>((c.z * dim.y + c.y) * dim.x + c.x);
}

__device__ __forceinline__
uint32_t lid(int3 c, int3 dim) { return linIdx(c, dim); }

#else

// 非 CUDA 编译环境（例如 .cpp 中的 CPU fallback）提供主机内联实现
inline bool inBounds(int3 c, int3 dim) {
    return (c.x >= 0 && c.x < dim.x &&
            c.y >= 0 && c.y < dim.y &&
            c.z >= 0 && c.z < dim.z);
}

inline uint32_t linIdx(int3 c, int3 dim) {
    return static_cast<uint32_t>((c.z * dim.y + c.y) * dim.x + c.x);
}

inline uint32_t lid(int3 c, int3 dim) { return linIdx(c, dim); }

#endif

} // namespace sim