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

// —— 压缩网格：在排序后的 unique keys 上二分查段 ——

// 返回 lower_bound(keys, keys+M, key)
__device__ __forceinline__
int lower_bound_u32(const uint32_t* keys, int M, uint32_t key) {
    int lo = 0, hi = M;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        uint32_t v = keys[mid];
        if (v < key) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// 若命中，输出 [beg,end)，否则返回 false
__device__ __forceinline__
bool compact_cell_range(
    const uint32_t* uniqueKeys,
    const uint32_t* offsets,
    uint32_t M,
    uint32_t key,
    uint32_t& beg,
    uint32_t& end)
{
    if (M == 0u) { beg = end = 0u; return false; }
    int pos = lower_bound_u32(uniqueKeys, int(M), key);
    if (pos >= int(M) || uniqueKeys[pos] != key) { beg = end = 0u; return false; }
    beg = offsets[pos];
    end = offsets[pos + 1];
    return (beg < end);
}

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